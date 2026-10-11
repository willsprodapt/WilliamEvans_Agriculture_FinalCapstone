# Sample test cases (real outputs)

Captured with `use_llm=false` (deterministic path, no API credit). With a funded `ANTHROPIC_API_KEY` the same queries also return a Claude explanation and judge verdict.

## Query
> Which crops perform well in Punjab during the Rabi season with soil pH 7.8, rainfall around 600 mm and nitrogen 150?

- status: `completed_with_warnings`  - duration: 16435 ms
- detected: `{'state': ('Punjab', 'parsed_from_query'), 'season': ('Rabi', 'parsed_from_query'), 'n': (150.0, 'parsed_from_query'), 'ph': (7.8, 'parsed_from_query'), 'rainfall': (600.0, 'parsed_from_query')}`
- **#1 Wheat** score 0.9355 | historical median yield 4.50 (n=23) | predicted 4.6861 (estimate) | risk low | cases TM-4F967CC8E7, TM-71FA4CA8FD, TM-AB8C52CA78
- **#2 Gram** score 0.9057 | historical median yield 1.05 (n=22) | predicted 1.2234 (estimate) | risk low | cases TM-CCCF09D177, TM-8104480866, TM-46540AC2FD
- **#3 Barley** score 0.8709 | historical median yield 3.30 (n=3) | predicted 3.6863 (estimate) | risk low | cases TM-F1FDDE76C9, TM-B34C1D93A4, TM-9678345FFB
- warning: LLM-as-judge did not run (not_run: disabled); only deterministic validation applies.

## Query
> Suggest alternatives to rice for Kharif in West Bengal with pH 6.2.

- status: `completed_with_warnings`  - duration: 618 ms
- detected: `{'state': ('West Bengal', 'parsed_from_query'), 'season': ('Kharif', 'parsed_from_query'), 'ph': (6.2, 'parsed_from_query')}`
- **#1 Other Kharif pulses** score 0.9724 | historical median yield 0.55 (n=22) | predicted 0.8073 (estimate) | risk low | cases TM-271B429A36, TM-2E06ACCEB5, TM-A770EDF9D4
- **#2 Jute** score 0.8953 | historical median yield 15.50 (n=6) | predicted 14.9947 (estimate) | risk low | cases TM-D8F0285849, TM-9ECAD75505, TM-EAE087E702
- **#3 Sunflower** score 0.8694 | historical median yield 1.23 (n=7) | predicted 1.1334 (estimate) | risk low | cases TM-D728ED70EC, TM-346FEE94E1, TM-3799D8D32B
- warning: LLM-as-judge did not run (not_run: disabled); only deterministic validation applies.

## Query
> I want to grow Coconut in Delhi. Is it a good choice?

- status: `completed_with_warnings`  - duration: 516 ms
- detected: `{'crop': ('Coconut', 'parsed_from_query'), 'state': ('Delhi', 'parsed_from_query')}`
- **#1 Rice** score 0.7328 | historical median yield 2.82 (n=14) | predicted 3.0485 (estimate) | risk low | cases TM-64A909E4AD, TM-5E32C62094, TM-71E093F8FD
- **#2 Bajra** score 0.7085 | historical median yield 1.90 (n=13) | predicted 2.3984 (estimate) | risk low | cases TM-D180492E74, TM-D13CBBC749, TM-C88147D0AA
- **#3 Maize** score 0.6941 | historical median yield 3.63 (n=18) | predicted 8.279 (estimate) | risk moderate | cases TM-9E01CD4717, TM-1DF9858D57, TM-7858D4C914
- warning: No historical records of Coconut match your constraints (state=['Delhi']); Coconut is not recommended on evidence. Alternatives below are supported by data.
- warning: LLM-as-judge did not run (not_run: disabled); only deterministic validation applies.

## Query
> What should I plant?

- status: `needs_clarification`  - duration: 2 ms
- detected: `{}`
- clarification: I could not find a crop, state, season or any soil/weather values in your question. Please add at least a state or season (e.g. 'Kharif in Assam') or some conditions such as pH, rainfall or N/P/K.
