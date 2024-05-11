# Mathematical Capabilities of Large Language Models

This project aims to evaluate existing prompting and in-context learning techniques for LLMs on arithmetic skills and math word problem (MWP) solving capabilities. The focus is on arithmetic tasks that a 10-year-old pupil must be able to solve. For this purpose, we use [the dataset we created for arithmetic operations and MWPs](https://github.com/RamonKaspar/MathDataset-ElementarySchool) (sampled from existing datasets).

## Implementation of Multiple Techniques

For detailed information on how the various techniques are implemented, as well as instructions on how to add new techniques, please refer to the README in the `techniques` folder.

## Benchmark Dataset

### Arithmetic

We use the arithmetic dataset versions and `arithmetic_100.csv` and 100 samples. It was sampled from the [Math-401 (Yuan et al., 2023)](https://arxiv.org/abs/2304.02015) and the [Mathematics Dataset (Saxton et al., 2019)](https://openreview.net/pdf?id=H1gR5iR5FX) datasets. You can find more informations [here](https://github.com/RamonKaspar/MathDataset-ElementarySchool).

Here are an example from the dataset (in JSON format):

```json
{
  "category": "Arithmetic",
  "subcategory": "mul",
  "question": "Work out 5 * 354.",
  "answer": "1770.0",
  "reasoning": null,
  "source": "Mathematics Dataset (Google DeepMind)"
}
```

### Math Word Problems (MWP)

We use the word problems dataset versions `wordProblems_100.csv` with 100 samples. It was sampled from the [SVAMP (Patel et al., 2021)](https://arxiv.org/abs/2103.07191), [AddSub (Hosseini et al., 2014)](https://aclanthology.org/D14-1058/) and [MultiArith (Roy et al., 2016)](https://arxiv.org/abs/1608.01413) datasets. You can find more informations [here](https://github.com/RamonKaspar/MathDataset-ElementarySchool).

Here are an example from the dataset (in JSON format):

```json
{
  "category": "Word Problems",
  "subcategory": "add_sub",
  "question": "Jessica spent $ 10.22 on a cat toy , and a cage cost her $ 11.73 . What was the total cost of Jessica 's purchases ? ",
  "answer": 21.95,
  "reasoning": "X = 10.22 + 11.73",
  "source": "AddSub"
}
```

### Geometry

We use the word problems dataset versions `geometry_100.csv` with 100 samples. It was sampled from the [MathQA Geometry (Amini et al., 2019)](https://allenai.org/data/lila) dataset. You can find more informations [here](https://github.com/RamonKaspar/MathDataset-ElementarySchool).

Here are an example from the dataset (in JSON format):

```json
{
  "category": "Geometry",
  "subcategory": "geometry",
  "question": "a metallic sheet is of rectangular shape with dimensions 48 m x 36 m . from each of its corners , a square is cut off so as to make an open box . if the length of the square is 3 m , the volume of the box ( in m 3 ) is :",
  "answer": 3780.0,
  "reasoning": "n0 = 48.0\nn1 = 36.0\nn2 = 3.0\nn3 = 3.0\nt0 = n2 * 2.0\nt1 = n0 - t0\nt2 = n1 - t0\nanswer = n2 * t1 * t2\nprint(answer)",
  "source": "MathQA_Geometry"
}
```

## Evaluated Techniques

We distinguish between these two approaches:

- **Few-shot prompting**: refers to providing the model with a small number of examples (shots) before asking it to perform a task. These examples serve as a guide for how the task should be completed.
- **Zero-shot prompting** means the model is given a task without any prior examples. It relies solely on its pre-existing training to infer how to handle the task.

This table gives an overview of all techniques we evaluated. The column `Paper` cites the original source, and `Implementation` is the source of the implementation we used.

<table border="1" style="border-collapse: collapse; width: 100%;">
    <caption>Overview of the implemented and benchmarked techniques.</caption>
    <thead>
        <tr>
            <th><strong>Name</strong></th>
            <th><strong>Paper</strong></th>
            <th><strong>Implementation</strong></th>
            <th><strong>Description</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GPT-3.5-Turbo Baseline (<code>Baseline.py</code>)</a></td>
            <td>TECHNICAL REPORT?</td>
            <td>Implemented on my own.</td>
            <td>TODO</td>
        </tr>
        <tr>
            <td>Chain-of-Thought (<code>CoT.py</code>)</a></td>
            <td>J. Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” arXiv, Jan. 10, 2023. Accessed: Mar. 22, 2024. [Online]. Available: http://arxiv.org/abs/2201.11903</td>
            <td>-</td>
            <td>TODO</td>
        </tr>
        <tr>
            <td>Program-aided Language Models (<code>PaL.py</code>)</a></td>
            <td>L. Gao et al., “PAL: Program-aided Language Models.” arXiv, Jan. 27, 2023. doi: 10.48550/arXiv.2211.10435.</td>
            <td>-</td>
            <td>TODO</td>
        </tr>
        <tr>
            <td>Role-Play Prompting (<code>RolePlay.py</code>)</a></td>
            <td>A. Kong et al., “Better Zero-Shot Reasoning with Role-Play Prompting.” arXiv, Mar. 13, 2024. Accessed: May 03, 2024. [Online]. Available: http://arxiv.org/abs/2308.07702</td>
            <td>-</td>
            <td>Role-play prompting involves instructing a language model to adopt a specific persona (i.e. a Math teacher) to tailor its responses accordingly.</td>
        </tr>
        <tr>
            <td>Declarative with SymPy (<code>DeclarativeSymPy.py</code>)</a></td>
            <td>J. He-Yueya, G. Poesia, R. E. Wang, and N. D. Goodman, “Solving Math Word Problems by Combining Language Models With Symbolic Solvers.” arXiv, Apr. 16, 2023. Accessed: May 02, 2024. [Online]. Available: http://arxiv.org/abs/2304.09102</td>
            <td><a href="https://github.com/joyheyueya/declarative-math-word-problem/tree/main">Author's implementation</a>(with slight adaptations)</td>
            <td>Uses the LLM to incrementally formalize problems into variables and equations, then employs a symbolic solver (Python SymPy) to compute solutions.</td>
        </tr>
</table>
