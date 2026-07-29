import time
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
import ee
ee.Initialize(project='ee-ciceromartinsjr')

while True:
    tasks = [t for t in ee.batch.Task.list()
             if t.config and 'Rosamarina' in (t.config.get('description') or '')
             and 'densify' in (t.config.get('description') or '')]
    if not tasks:
        print('no matching tasks found'); break
    states = [t.status()['state'] for t in tasks]
    done = sum(s in ('COMPLETED', 'FAILED', 'CANCELLED') for s in states)
    print(f'{time.strftime("%H:%M:%S")}  {done}/{len(states)} finished  '
          f'({", ".join(sorted(set(states)))})')
    if done == len(states):
        for t in tasks:
            st = t.status()
            if st['state'] != 'COMPLETED':
                print('  NOT COMPLETED:', st.get('description'), st['state'], st.get('error_message'))
        break
    time.sleep(20)
