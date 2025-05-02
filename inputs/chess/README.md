# Chess Experiment Dataset

## Overview
This directory contains the data for the chess experiments in the paper from [Reasoning & Reciging](https://github.com/ZhaofengWu/counterfactual-evaluation/tree/master/chess). In this experiment, you ask LLMs whether a chess opening is legal. In counterfactual variants, you swap the initial position of knights and bishops.

## A note on the data
If you use the data directly, the file suffix _{T,F}_{T,F} indicates if the opening is legal in regular/counterfactual chess, respectively. The _T_T and _F_F files are empty because you want to make the test discriminative between default vs. counterfactual chess.
