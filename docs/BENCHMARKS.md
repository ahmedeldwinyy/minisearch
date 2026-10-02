# MiniSearch Benchmarks

SciFact corpus: 5,183 documents. Query averages use 100 runs.

## Measurements

| Operation | Measured time |
| --- | ---: |
| Index build (1,000 documents) | 0.174 s |
| Index build (all 5,183 documents) | 0.730 s |
| Single-term query (100 runs) | 0.069 ms/query |
| AND query (100 runs) | 0.151 ms/query |
| OR query (100 runs) | 0.164 ms/query |
| NOT query (100 runs) | 0.710 ms/query |
| Nested query (100 runs) | 1.210 ms/query |
| Trie build (5,183 documents) | 0.097 s |
| Trie prefix lookup (`co`, 100 runs) | 0.009 ms/query |

## Big-O

| Operation | Complexity | Measured time |
| --- | --- | ---: |
| Index build (1,000 documents) | O(T) expected | 0.174 s |
| Index build (all documents) | O(T) expected | 0.730 s |
| AND postings merge | O(|A| + |B|) | 0.151 ms/query |
| OR postings merge | O(|A| + |B|) | 0.164 ms/query |
| NOT postings merge | O(|A| + |U|) | 0.710 ms/query |
| Trie prefix lookup | O(|prefix| + V log sigma) | 0.009 ms/query for prefix `co` over 100 runs |

T is the number of indexed tokens, A and B are postings lists, U is the document universe, V is the trie nodes visited while finding completions, and sigma is the character alphabet size.
