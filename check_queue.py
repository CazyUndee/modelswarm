from modelswarm.client import Client

c = Client()

# Check for unclaimed experiments
exps = c.get_experiments(competition_id='playground-series-s6e8', status='pending')
print(f'Pending experiments: {len(exps)}')
for e in exps:
    print(f"  {e['experiment_id']} | {e.get('title', e.get('hypothesis', '?'))}")

print()

# Check for any unclaimed experiments
exps_all = c.get_experiments(competition_id='playground-series-s6e8')
unclaimed = [e for e in exps_all if e.get('status') == 'pending' or e.get('claimed_by') is None]
print(f'Unclaimed: {len(unclaimed)}')
for e in unclaimed:
    print(f"  {e['experiment_id']} | status={e['status']} | claimed_by={e.get('claimed_by')}")
