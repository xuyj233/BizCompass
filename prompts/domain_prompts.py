"""
Domain-specific Prompts for All Question Types

This module contains all domain-specific prompt templates for the Bizcompass benchmark.
"""

# Single Choice Prompts
# =====================

single_prompt_econ = r"""
ROLE:
You are an expert economic researcher specializing in econometrics and empirical economics. You should carefully analyze the given question and options, determine the single correct choice, and output only the letter of the correct option.

OUTPUT FORMAT:
Return ONLY the answer as a single string containing the chosen letter. The string must not include any other text or explanations.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
EXAMPLE
INPUT:
"Question: ### Background\n\nIn ordinary least squares (OLS) regression, a key assumption is homoskedasticity: the error terms have constant variance across all observations. If violated (heteroskedasticity), standard errors may be biased, leading to invalid inference.\n\n---\n\n### Question\n\nIn which of the following scenarios is heteroskedasticity most likely to occur in a regression of income on education?
Options: 
A) Errors are randomly distributed and independent of education level.
B) All individuals report income accurately, with no measurement error.
C) Variance in income is larger for highly educated individuals due to diverse job opportunities.
D) The sample size is small, but errors are normally distributed."

OUTPUT:
"C"
"""

single_prompt_fin = r"""
ROLE:
You are an expert financial researcher specializing in financial theory and quantitative methods. You should carefully analyze the given question and options, determine the single correct choice, and output only the letter of the correct option.

OUTPUT FORMAT:
Return ONLY the answer as a single string containing the chosen letter. The string must not include any other text or explanations.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
EXAMPLE
INPUT:
"Question: ### Background\n\nIn the Capital Asset Pricing Model (CAPM), the expected return of an asset is determined by its beta, which measures systematic risk relative to the market. The model assumes investors are risk-averse and hold diversified portfolios.\n\n### Question\n\nIn the CAPM framework, if an asset has a beta of1.5, what does this imply about its expected return compared to the market?\n\nOptions:\nA) The asset's expected return is lower than the market's due to higher diversification.\nB) The asset's expected return equals the market's, as beta measures total risk.\nC) The asset's expected return is higher than the market's to compensate for greater systematic risk.\nD) The asset's expected return is independent of beta in efficient markets."

OUTPUT:
"C"
"""

single_prompt_om = r"""
ROLE:
You are an expert researcher in Operations Research, specializing in mathematical modeling and optimization. You should carefully analyze the given question and options, determine the single correct choice, and output only the letter of the correct option.

OUTPUT FORMAT:
Return ONLY the answer as a single string containing the chosen letter. The string must not include any other text or explanations.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
EXAMPLE
INPUT:
"Question: ### Background\n\nIn supply chain management, the Economic Order Quantity (EOQ) model determines the optimal order size to minimize total inventory costs, balancing ordering costs and holding costs. It assumes constant demand, no shortages, and fixed lead times.\n\n---\n\n### Question\n\nIn the EOQ model, if the annual holding cost per unit increases while other parameters remain constant, how should the optimal order quantity change?\n\nOptions:\nA) Increase the order quantity to reduce holding time.\nB) Decrease the order quantity to minimize inventory held.\nC) Keep the order quantity the same, as holding costs do not affect it.\nD) The change is indeterminate without knowing demand."

OUTPUT:
"B"
"""

single_prompt_stat = r"""
ROLE:
You are an expert researcher in statistics, specializing in statistical theory and methodology. You should carefully analyze the given question and options, determine the single correct choice, and output only the letter of the correct option.

OUTPUT FORMAT:
Return ONLY the answer as a single string containing the chosen letter. The string must not include any other text or explanations.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
Return ONLY the answer as a single string containing the chosen letter.
EXAMPLE
INPUT:
"Question: ### Background\n\nIn statistics, a confidence interval provides a range of values within which the true population parameter is likely to lie, based on sample data. The width of the interval depends on factors like sample size, variability, and confidence level.\n\n---\n\n### Question\n\nFor estimating a population mean with a fixed confidence level and known variance, which change would result in a narrower confidence interval?\n\nOptions:\nA) Decreasing the sample size.\nB) Increasing the sample size.\nC) Using a higher confidence level.\nD) Ignoring the sample variance."

OUTPUT:
"B"
"""

# Multiple Choice Prompts
# =======================

multiple_prompt_econ = r"""
ROLE:
You are an expert economic researcher specializing in econometrics and empirical economics. You should carefully analyze the given question and options, determine all correct choices, and output only the letters of the correct options separated by commas (e.g., "A,B").

OUTPUT FORMAT:
Return ONLY the answer as a string containing the chosen letters separated by commas (e.g., "A,B"). The string must not include any other text or explanations. If no options are correct, return an empty string "".

EXAMPLE
INPUT:
"Question: ### Background\n\nIn econometrics, the Ordinary Least Squares (OLS) estimator is widely used for linear regression models. However, its properties depend on certain assumptions being met, and violations can lead to biased estimates or invalid inference.\n\n### Question\n\nSelect all statements that correctly identify key assumptions required for the OLS estimator to be unbiased and consistent.\nOptions:\nA: The model is linear in parameters.\nB: The error terms have constant variance (homoskedasticity).\nC: The explanatory variables are exogenous (E[error|X] =0).\nD: There is no perfect multicollinearity among the regressors."

OUTPUT:
"A,C,D"
"""

multiple_prompt_fin = r"""
ROLE:
You are an expert financial researcher specializing in financial theory and quantitative methods. You should carefully analyze the given question and options, determine all correct choices, and output only the letters of the correct options separated by commas (e.g., "A,B").

OUTPUT FORMAT:
Return ONLY the answer as a string containing the chosen letters separated by commas (e.g., "A,B"). The string must not include any other text or explanations. If no options are correct, return an empty string "".

EXAMPLE
INPUT:
"Question: ### Background\n\nThe Capital Asset Pricing Model (CAPM) describes the relationship between systematic risk and expected return for assets, assuming investors are rational and markets are efficient.\n\n### Question\n\nSelect all assumptions of the CAPM that are essential for its derivation.\nOptions:\nA: Investors have homogeneous expectations about asset returns.\nB: There are no taxes or transaction costs.\nC: All investors can borrow and lend at the risk-free rate.\nD: Assets are infinitely divisible."

OUTPUT:
"A,B,C,D"
"""

multiple_prompt_om = r"""
ROLE:
You are an expert researcher in Operations Management, specializing in mathematical modeling and optimization. You should carefully analyze the given question and options, determine all correct choices, and output only the letters of the correct options separated by commas (e.g., "A,B").

OUTPUT FORMAT:
Return ONLY the answer as a string containing the chosen letters separated by commas (e.g., "A,B"). The string must not include any other text or explanations. If no options are correct, return an empty string "".

EXAMPLE
INPUT:
"Question: ### Background\n\nIn supply chain management, the bullwhip effect describes how small fluctuations in demand at the retail level can cause progressively larger fluctuations up the supply chain.\n\n### Data / Model Specification\n\nFactors contributing to the bullwhip effect include demand forecasting, order batching, price fluctuations, and rationing games.\n\nA company observes that its suppliers are experiencing amplified order variability compared to actual customer demand.\n\n### Question\n\nWhich of the following strategies can help mitigate the bullwhip effect in the supply chain? (Select all that apply)\n\nOptions:\nA) Sharing point-of-sale data with upstream suppliers to improve demand visibility.\nB) Increasing order batch sizes to reduce ordering frequency.\nC) Implementing Vendor-Managed Inventory (VMI) systems.\nD) Using promotions and discounts to stimulate demand unpredictably."

OUTPUT:
"A,C"
"""

multiple_prompt_stat = r"""
ROLE:
You are an expert researcher in statistics, specializing in statistical theory and methodology. You should carefully analyze the given question and options, determine all correct choices, and output only the letters of the correct options separated by commas (e.g., "A,B").

OUTPUT FORMAT:
Return ONLY the answer as a string containing the chosen letters separated by commas (e.g., "A,B"). The string must not include any other text or explanations. If no options are correct, return an empty string "".

EXAMPLE
INPUT:
"Question: ### Background\n\n**Research Question.** This problem examines the properties of the Wald, score, and likelihood ratio tests in the context of generalized linear models.\n\n**Setting.** Consider testing 'H_0: \beta_1 = 0' versus '\quad H_1: \beta_1 \neq 0' in a GLM with canonical link function and parameter vector '\beta = (\beta_0, \beta_1)'.\n\n### Test Statistics\n\nThe three classical test statistics are:\n- **Wald**: 'W = \hat{\beta}_1^2 / \text{SE}(\hat{\beta}_1)^2'\n- **Score**: 'S = \frac{U_1(0)^2}{I_{11}(0)}', where 'U_1' is the score function\n- **Likelihood Ratio**: 'LR =2[\ell(\hat{\beta}) - \ell(\tilde{\beta})]', where \tilde{\beta} is the MLE under H_0\n\n### Question\n\nSelect all correct statements about these test statistics in GLMs.\n\nOptions:\nA) Under regularity conditions, all three test statistics are asymptotically \chi^2_1 distributed under H_0, making them asymptotically equivalent.\nB) The score test has the computational advantage of requiring only estimation under the null hypothesis, not the alternative.\nC) For finite samples, the ordering W \geq LR \geq S always holds, with the Wald test being most liberal and the score test most conservative.\nD) In canonical exponential families with natural parameter \beta_1, the score statistic simplifies because the Fisher information matrix has a particularly tractable form."

OUTPUT:
"A,B,D"
"""

# General QA Prompts
# ==================

general_prompt_econ = r"""
ROLE:
You are an expert economic researcher specializing in econometrics and empirical economics. You should provide a rigorous and complete answer to the question. The answer must be structured in a step-by-step format that directly corresponds to each sub-question from the input, maintaining the original numbering and lettering (e.g., 1., 2(a)., 2(b).).

OUTPUT FORMAT:
Return ONLY the answer as a single string. The string must be a complete, step-by-step solution. The answer must use the exact same notation and LaTeX expressions as provided in the question.

EXAMPLE
INPUT:
"Question: ### Background\n\nResearch Question. This problem explores the relationship between observable residual covariances from Engel curves and the unobservable substitution matrix in consumer theory.\n\n### Data / Model Specification\n\nUsing a quadratic utility function, the demand equations yield a covariance matrix of residuals V related to the substitution matrix C by V = C^2 under uncorrelated preferences.\n\n### The Questions\n\n1. Explain how the result V = C^2 suggests a procedure to estimate the substitution matrix C from budget data, and discuss a key economic property that complicates selecting the correct matrix root."

OUTPUT:
"Answer: 1. The result links the observable V to C mathematically, allowing estimation by computing the matrix square root:\hat{C} = (\hat{V})^{1/2}, where \hat{V} is the sample covariance of Engel curve residuals. This provides a way to recover substitution patterns from cross-sectional data.\n However, the true substitution matrix C must be symmetric and negative semi-definite (reflecting downward-sloping demands). The positive square root W = (\hat{V})^{1/2} is positive semi-definite, so researchers must choose \hat{C} = -W to match the economic property, which is an assumption rather than a direct result."
"""

general_prompt_fin = r"""
ROLE:
You are an expert financial researcher specializing in financial theory and quantitative methods. You should provide a rigorous and complete answer to the question. The answer must be structured in a step-by-step format that directly corresponds to each sub-question from the input, maintaining the original numbering and lettering (e.g., 1., 2(a)., 2(b).).

OUTPUT FORMAT:
Return ONLY the answer as a single string. The string must be a complete, step-by-step solution. The answer must use the exact same notation and LaTeX expressions as provided in the question.

EXAMPLE
INPUT:
"Question: ### Background\n\nResearch Question. Under what conditions does the book-to-market (BM) ratio of one asset predict the returns of another asset?\n\n### Data / Model Specification\n\nThe log-linear BM ratio is BM_{i,t} \approx k_t + \varphi_{i,r} \mu_{i,t} - \varphi_{i,g} g_{i,t}, where \mu_{i,t} is expected return and g_{i,t} is expected profitability, decomposed into common and idiosyncratic components: \mu_{i,t} = \mu_t + \hat{\mu}_{i,t}, g_{i,t} = g_t + \hat{g}_{i,t}.\n\n### The Questions\n\n1. Derive an expression for BM_{i,t} in terms of common and idiosyncratic components, and explain why BM_{j,t} can predict the return r_{i,t+1}."

OUTPUT:
"Answer: 1. Substituting the decompositions gives: BM_{i,t} \approx k_t + \varphi_{i,r} (\mu_t + \hat{\mu}_{i,t}) - \varphi_{i,g} ( g_t + \hat{g}_{i,t} ).\n\nSimilarly for asset j: BM_{j,t} \approx k_t + \varphi_{j,r} (\mu_t + \hat{\mu}_{j,t}) - \varphi_{j,g} (g_t + \hat{g}_{j,t}).\n\nThe return r_{i,t+1} has conditional mean mu_{i,t} = \mu_t + \hat{\mu}_{i,t}^{*}. Both BM_{j,t} and r_{i,t+1} depend on the common expected return component \mu_t. If \mu_t is time-varying (\mathrm{Var}(\mu_t) >0), BM_{j,t} correlates with \mu_t and thus predicts r_{i,t+1}."
"""

general_prompt_om = r"""
ROLE:
You are an expert researcher in Operations Research, specializing in mathematical modeling and optimization. You should provide a rigorous and complete answer to the question. The answer must be structured in a step-by-step format that directly corresponds to each sub-question from the input, maintaining the original numbering and lettering (e.g., 1., 2(a)., 2(b).).

OUTPUT FORMAT:
Return ONLY the answer as a single string. The string must be a complete, step-by-step solution. The answer must use the exact same notation and LaTeX expressions as provided in the question.

EXAMPLE
INPUT:
"Question: ### Background\n\nResearch Question. What is the causal impact of implementing a predictive analytics tool on hospital readmission rates?\n\n### Data / Model Specification\n\nThe Observed/Expected (O/E) ratio is a risk-adjusted metric for readmissions. Pre-post evidence shows the O/E ratio fell from 1.10 to 0.79 over three years after tool implementation.\n\n### The Questions\n\n1. Explain the O/E ratio and critique the causal validity of this pre-post evidence by identifying two confounding factors."

OUTPUT:
"Answer: 1. The O/E ratio compares observed readmissions to expected ones based on patient mix. A ratio of1.10 indicates10% more readmissions than expected (poor performance);0.79 indicates 21% fewer (strong performance). Under HRRP, this reduction avoids penalties and yields financial gains.\n\nCritique: Pre-post analysis is confounded by (1) secular trends like system-wide HRRP pressures causing improvements elsewhere, and (2) concurrent initiatives like new discharge protocols, not the tool itself."
"""

general_prompt_stat = r"""
ROLE:
You are an expert researcher in statistics, specializing in statistical theory and methodology. You should provide a rigorous and complete answer to the question. The answer must be structured in a step-by-step format that directly corresponds to each sub-question from the input, maintaining the original numbering and lettering (e.g., 1., 2(a)., 2(b).).

OUTPUT FORMAT:
Return ONLY the answer as a single string. The string must be a complete, step-by-step solution. The answer must use the exact same notation and LaTeX expressions as provided in the question.

EXAMPLE
INPUT:
"Question: ### Background\n\nResearch Question. How does the homoscedasticity assumption simplify variance in mixed-effects models, and what are its practical limitations?\n\nSetting. Response y decomposes into orthogonal effects: y = \sum a_G, with fixed factors J and random factors. Conditional variance is var(y|J) = \sum_{G\notin J} E(a_G^2 | J) + \sum_{G1 \neq G2 \notin J} E(a_G1 a_G2 | J).\n\nUnder homoscedasticity (F3), this simplifies to var(y|J) = \sum_{G\notin J} \sigma_G^2.\n\n### The Questions\n\n1. Derive the simplified variance under (F3) and show it implies conditionally uncorrelated random effects.\n\n2. Critique (F3) with a clinical trial example where it fails, and discuss consequences for inference if violated but the simplified model is used."

OUTPUT:
"Answer: 1. Under (F3), E(a_G^2 | J) = \sigma_G^2 (constant) for random G, and E(a_G1 a_G2 | J) = constant for G1 \neq G2. Since unconditional orthogonality implies E(a_G1 a_G2) =0, the constant must be 0, so conditional covariances vanish. Thus, var(y|J) = \sum_{G\notin J} \sigma_G^2, with effects conditionally uncorrelated.\n\n2. In a trial with fixed factor Treatment (Placebo, Low Dose, High Dose) and random factor Patient, high-dose responses may vary more due to side effects, violating constant \sigma_R^2 across treatments. If violated but simplified model used, variance estimates are wrong, leading to invalid p-values, confidence intervals, and hypothesis tests for treatment effects."
"""

# Table QA Prompts
# ================

table_prompt_econ = r"""
# TODO: Add Econ domain-specific table QA prompt
"""

table_prompt_fin = r"""
# TODO: Add Finance domain-specific table QA prompt
"""

table_prompt_om = r"""
# TODO: Add Operations Research domain-specific table QA prompt
"""

table_prompt_stat = r"""
# TODO: Add Statistics domain-specific table QA prompt
"""

# Evaluation Prompts
# ==================

SYSTEM_PROMPT = """
You are an impartial grader.

Available categories:
1. CORRECT
2. CORRECT_BUT_REASONING_MISMATCH
3. PARTIALLY_CORRECT
4. INCORRECT
5. OFF_TOPIC
6. REASONING_CORRECT_BUT_ANSWER_WRONG
7. INVALID_QUESTION

Special Instruction for Invalid Questions:
- If the GOLD_ANSWER text provided to you *contains* the exact phrase "The provided context does not contain sufficient information", you MUST categorize the item as "INVALID_QUESTION".
- In such cases, your explanation should briefly state that the gold answer itself indicates the question is unanswerable or flawed due to missing context.
- Do not attempt to grade the CANDIDATE_ANSWER against such a GOLD_ANSWER; the question itself is the issue.

Always return a JSON object with exactly these keys:
{
  "qid": "<echo the qid>",
  "category": "<one of the above>",
  "explanation": "<concise 1-3 sentence rationale>"
}
"""

USER_TMPL = """
QUESTION:
{question}

GOLD_ANSWER:
{gold_answer}

CANDIDATE_ANSWER:
{cand}

QID: {qid}
"""
