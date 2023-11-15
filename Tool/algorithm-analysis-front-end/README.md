# Visualization 

In order to better observe the MTS data, we developed one visualization tool(under folder ``algorithm-analysis-front-end`` ).
After clone the repo, you need to 

```bash
npm install
npm run dev
```

then you need to put the _XXX.json_ file you want to visualize under folder ``algorithm-analysis-front-end/public``


The _XXX.json_ file are supposed to organized like:

| Key\_1             | Key\_2       | Value/Key\_3                                                  | Value                                                        |
| ----------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| '1'(entity index) | data        | *n* (*n* = number of entities) lists, each is *metric_num\*timestamp*. Take `` application-server-dataset.json`` for instance it would be [[4320\*19],[4320\*19],...,[4320\*19]] | -                                                            |
|                   | label       | *n* (*n* = number of entities) lists, each is *timestamp* length. Take `` application-server-dataset.json`` for instance it would be [[4320],[4320],...,[4320]] | -                                                            |
|                   | algorithm\_a | score                                                        | Anomaly score of algorithm_a on this entity , which is *timestamp* length. For example it would be [3212.12, 43.212,..., -2143.34] |
|                   |             | threshold                                                    | the threshold of anomaly score. For example it would be like -217.45. |
|                   |             | p                                                            | precision of the algorithom. For example it would be like 0.8. |
|                   |             | r                                                            | recall of the algorithom. For example it would be like 0.6.  |
|                   |             | f1                                                           | f1 of the algorithom. For example it would be like 0.686.    |
|                   | algorithm\_b | score                                                        | [412.3, 63.612,..., 83.9]                                    |
|                   |             | threshold                                                    | -17.45.                                                      |
|                   |             | p                                                            | 0.6                                                          |
|                   |             | r                                                            | 0.8                                                          |
|                   |             | f1                                                           | 0.686                                                        |
|                   | ...         | ...                                                          | ...                                                          |
|                   | algorithm\_x | ...                                                          | ...                                                          |
| '2'               | ...         | ...                                                          | ...                                                          |
| ...               | ...         | ...                                                          | ...                                                          |
| '11'              | ...         | ...                                                          | ...                                                          |