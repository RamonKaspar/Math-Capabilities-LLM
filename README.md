# Mathematical Capabilities of Large Language Models

This project aims to evaluate existing prompting and in-context learning techniques for LLMs on arithmetic skills and math word problem (MWP) solving capabilities. The focus is on arithmetic tasks that a 10-year-old pupil must be able to solve. For this purpose, we use [the dataset we created for arithmetic operations and MWPs](https://github.com/RamonKaspar/MathDataset-ElementarySchool) (sampled from existing datasets).

## Implementation of Multiple Techniques

For detailed information on how the various techniques are implemented, as well as instructions on how to add new techniques, please refer to the README in the `techniques` folder.

## Benchmark Dataset

### Arithmetic

We use the arithmetic dataset versions `arithemtic_1000.csv` and `arithmetic_100.csv` with 1000 and 100 samples, respectively. It was sampled from the [Math-401 (Yuan et al., 2023)](https://arxiv.org/abs/2304.02015) and the [Mathematics Dataset (Saxton et al., 2019)](https://openreview.net/pdf?id=H1gR5iR5FX) datasets. You can find more informations [here](https://github.com/RamonKaspar/MathDataset-ElementarySchool).

Here are a few examples from the dataset (in JSON format):

```json
{"category":"Arithmetic","subcategory":"mul","question":"Work out 5 * 354.","answer":"1770.0","reasoning":null,"source":"Mathematics Dataset (Google DeepMind)"}
{"category":"Arithmetic","subcategory":"time","question":"How many minutes are there between 6:56 PM and 5:27 AM?","answer":"631","reasoning":null,"source":"Mathematics Dataset (Google DeepMind)"}
{"category":"Arithmetic","subcategory":"arithmetic_mixed","question":"4-10=","answer":"-6.0","reasoning":null,"source":"Math-401"}
```

### Math Word Problems (MWP)

We use the word problems dataset versions `wordProblems_1000.csv` and `wordProblems_100.csv` with 1000 and 100 samples, respectively. It was sampled from the [SVAMP (Patel et al., 2021)](https://arxiv.org/abs/2103.07191), [AddSub (Hosseini et al., 2014)](https://aclanthology.org/D14-1058/) and [MultiArith (Roy et al., 2016)](https://arxiv.org/abs/1608.01413) datasets. You can find more informations [here](https://github.com/RamonKaspar/MathDataset-ElementarySchool).

Here are a few examples from the dataset (in JSON format):

```json
{"category":"Word Problems","subcategory":"add_sub","question":"Jessica spent $ 10.22 on a cat toy , and a cage cost her $ 11.73 . What was the total cost of Jessica 's purchases ? ","answer":21.95,"reasoning":"X = 10.22 + 11.73","source":"AddSub"}
{"category":"Word Problems","subcategory":"multi_step","question":" At the schools book fair Sam bought 13 adventure books and 17 mystery books. If 15 of the books were used, how many new books did he buy? ","answer":15.0,"reasoning":"X=((13.0+17.0)-15.0)","source":"MultiArith"}
```

## Evaluated Techniques

This table gives an overview of all techniques we evaluated. The column `Paper` cites the original source, and `Implementation` is the source of the implementation we used.

<table border="1" style="border-collapse: collapse; width: 100%;">
    <caption>Overview of the implemented and benchmarked techniques.</caption>
    <thead>
        <tr>
            <th><strong>Name</strong></th>
            <th><strong>Paper</strong></th>
            <th><strong>Implementation</strong></th>
            <th><strong>Licence</strong></th>
            <th><strong>Description</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GPT-3.5-Turbo Baseline</a></td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>We used the single instrution "Just return the answer to the problem." and the few-shot solutions are just the numbers.</td>
        </tr>
        <tr>
            <td><a href="https://github.com/GanjinZero/math401-llm">Math-401</a></td>
            <td><code>arithmetic_mixed</code></td>
            <td>63</td>
            <td>log 10(797)=</td>
        </tr>
</table>
