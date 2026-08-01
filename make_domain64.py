import json
from pathlib import Path
base=json.loads(Path('domain_prompts.json').read_text())
extra=[
 'translate English to German: The birds are near the house.',
 'translate English to French: The river crosses the city.',
 'translate English to Spanish: The child opens the window.',
 'translate English to Italian: The book is on the desk.',
 'summarize: The laboratory recorded a signal during the evening experiment.',
 'summarize: Heavy rain delayed the train and closed the road.',
 'summarize: The doctor explained the treatment to the patient.',
 'summarize: Volunteers cleaned the park after the festival.',
 'question: What do bees make? answer:',
 'question: Which animal barks? answer:',
 'question: What season follows spring? answer:',
 'question: How many legs does a spider have? answer:',
 'classify sentiment: The thoughtful design made the task pleasant.',
 'classify sentiment: The broken device caused a frustrating delay.',
 'complete: The small boat moved across the',
 'complete: A careful student wrote a'
]
Path('domain_prompts64.json').write_text(json.dumps(base+extra,indent=2)+'\n'); print({'prompts':len(base+extra),'output':'domain_prompts64.json'})
