"""Rebuild historical S&P 500 membership from Wikipedia page revisions.

Wikipedia's "List of S&P 500 companies" no longer carries the changes table,
so this reads the last revision that did (REV_TS) and walks the changes
backwards from that revision's constituent list. Changes after REV_TS are
taken from the current page's "Date added" column.

Run: python src/research/sp500_membership.py
Writes output/research/sp500_membership.csv (ticker,start,end; end blank = still in).

Limits: Wikipedia's change list is community-maintained and thins out before
about 2008; ticker renames can leave a company under two symbols.
"""
import io
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'research' / 'sp500_membership.csv'
API = 'https://en.wikipedia.org/w/api.php'
HEAD = {'User-Agent': 'Mozilla/5.0 (research script)'}
REV_TS = '2026-06-01T00:00:00Z'
FLOOR = pd.Timestamp('1990-01-01')


def tables(ts=None):
    if ts is None:
        html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=HEAD, timeout=30).text
        return pd.Timestamp.now().normalize(), pd.read_html(io.StringIO(html))
    q = requests.get(API, headers=HEAD, timeout=30, params=dict(
        action='query', prop='revisions', titles='List of S&P 500 companies', rvlimit=1,
        rvstart=ts, rvdir='older', rvprop='ids|timestamp', format='json')).json()
    rev = list(q['query']['pages'].values())[0]['revisions'][0]
    html = requests.get(API, headers=HEAD, timeout=60, params=dict(
        action='parse', oldid=rev['revid'], prop='text', format='json')).json()['parse']['text']['*']
    return pd.Timestamp(rev['timestamp']).tz_localize(None).normalize(), pd.read_html(io.StringIO(html))


def clean(s):
    return str(s).strip().upper().replace('.', '-')


def main():
    rev_date, old = tables(REV_TS)
    members = {clean(s) for s in old[0]['Symbol']}
    ch = old[1].copy()
    ch.columns = ['date', 'added', 'added_name', 'removed', 'removed_name', 'reason']
    ch['date'] = pd.to_datetime(ch['date'], errors='coerce')
    ch = ch.dropna(subset=['date']).sort_values('date', ascending=False)

    intervals = []                      # (ticker, start, end)
    open_end = {t: pd.NaT for t in members}   # end date of the interval we are walking back through
    odd = 0
    for date, grp in ch.groupby('date', sort=False):
        for _, r in grp.iterrows():
            a, d = r['added'], r['removed']
            if isinstance(a, str) and a.strip():
                a = clean(a)
                if a in open_end:
                    intervals.append((a, date, open_end.pop(a)))
                else:
                    odd += 1
            if isinstance(d, str) and d.strip():
                d = clean(d)
                if d not in open_end:
                    open_end[d] = date
                else:
                    odd += 1
    for t, end in open_end.items():     # members before the first recorded change
        intervals.append((t, FLOOR, end))
    M = pd.DataFrame(intervals, columns=['ticker', 'start', 'end'])

    # Changes after the old revision: compare with the current page.
    _, cur = tables()
    now = {clean(s): pd.to_datetime(d, errors='coerce') for s, d in zip(cur[0]['Symbol'], cur[0]['Date added'])}
    added = sorted((d, t) for t, d in now.items() if t not in members)
    gone = sorted(members - set(now))
    for i, t in enumerate(gone):        # removal date unknown: pair with the i-th addition
        when = added[min(i, len(added) - 1)][0] if added else rev_date
        M.loc[(M.ticker == t) & M.end.isna(), 'end'] = when if pd.notna(when) else rev_date
    for d, t in added:
        M.loc[len(M)] = (t, d if pd.notna(d) else rev_date, pd.NaT)

    M = M.sort_values(['ticker', 'start']).reset_index(drop=True)
    M.to_csv(OUT, index=False)
    print(f'{len(M)} intervals, {M.ticker.nunique()} tickers, {len(ch)} changes back to {ch.date.min().date()}, '
          f'{odd} inconsistent rows, post-revision adds={len(added)} removes={len(gone)}')
    for y in (2005, 2008, 2012, 2016, 2020, 2024, 2026):
        d = pd.Timestamp(f'{y}-06-30')
        print(y, int(((M.start <= d) & (M.end.isna() | (M.end > d))).sum()), 'members; changes that year:', int((ch.date.dt.year == y).sum()))


if __name__ == '__main__':
    main()
