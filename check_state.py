from modelswarm.client import Client

c = Client()

print('=== Competition State ===')
state = c.get_competition_state('playground-series-s6e8')
for k, v in state.items():
    print(f'  {k}: {v}')

print()
print('=== Experiments ===')
exps = c.get_experiments(competition_id='playground-series-s6e8')
for e in exps:
    eid = e.get('id', '?')
    status = e.get('status', '?')
    title = e.get('title', '?')
    agent = e.get('agent_id', '?')
    print(f'  {eid} | {status} | {title} | agent={agent}')

print()
print('=== Agents ===')
agents = c.get_agents()
for a in agents:
    aid = a.get('id', '?')
    name = a.get('name', '?')
    status = a.get('status', '?')
    print(f'  {aid} | {name} | {status}')

print()
print('=== My Agent ===')
print(c.whoami())
