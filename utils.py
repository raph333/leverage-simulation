import os
from typing import List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://datahub.io/core/s-and-p-500/r/0.csv"
TMP_FILE = "tmp.csv"


@dataclass
class Params:
    n_years: int
    start_capital: int
    leverage_values: Tuple[float]
    interest_percent: float
    n_simulations: int


def set_globals(**kwargs):
    global PARAMS
    PARAMS = Params(**kwargs)


def get_data() -> pd.DataFrame:
    if not os.path.exists(TMP_FILE):
        os.system(f"curl -L {DATA_URL} > {TMP_FILE}")
    df = pd.read_csv("tmp.csv")
    df = df[["Date", "SP500"]].rename({"SP500": "Price"}, axis=1)
    df.Date = pd.to_datetime(df.Date)
    return df


def get_random_time_span(df: pd.DataFrame) -> pd.DataFrame:
    last = df.Date.max()
    last_possible = last - relativedelta(years=PARAMS.n_years)
    start = df[df.Date < last_possible].sample(1).Date.item()
    end = start + relativedelta(years=PARAMS.n_years)
    return df[(df.Date >= start) & (df.Date <= end)]


def add_multiple(df: pd.DataFrame) -> pd.DataFrame:
    multiples = [1]
    balance = df.iloc[0].Price

    for _, row in df.iloc[1:].iterrows():
        end = row.Price

        daily_return = (end - balance) / balance
        multiple = 1 + daily_return
        multiples.append(multiple)

        balance = balance * multiple

    df["multiple"] = multiples
    return df


def calculate_returns(df: pd.DataFrame, leverage: float = 1) -> List[float]:
    running_balance = []
    balance = PARAMS.start_capital
    running_balance.append(balance)

    for _, row in df.iloc[1:].iterrows():
        gain = leverage * (row.multiple - 1)
        leveraged_multiple = 1 + gain

        debt = balance * max((leverage - 1), 0)
        interest_amount = (debt * (PARAMS.interest_percent / 100)) / 365

        balance = max(balance * leveraged_multiple - interest_amount, 0)
        running_balance.append(balance)

    return running_balance


def run_simulation(df: pd.DataFrame, y_max_quantile_limit: Optional[float] = None):
    results = defaultdict(list)

    for _ in range(PARAMS.n_simulations):
        time_frame = get_random_time_span(df)
        for leverage in set(PARAMS.leverage_values).union([1]):
            multiple = (
                calculate_returns(time_frame, leverage=leverage)[-1]
                / PARAMS.start_capital
            )
            results[leverage].append(multiple)

    results_df = pd.DataFrame(results).melt(var_name="leverage", value_name="multiple")

    plot_outcome_distribution(
        results_df,
        y_limit=results[max(PARAMS.leverage_values)] if y_max_quantile_limit else None,
    )
    plot_minimum_multiples(results_df)
    plot_fraction_of_outcomes_worse_than_reference(results)


def plot_outcome_distribution(results_df: pd.DataFrame, y_limit: Optional[float]):
    fig, ax = plt.subplots()
    sns.boxplot(x="leverage", y="multiple", data=results_df, ax=ax)
    ax.set_title(f"Distributions of outcomes after {PARAMS.n_years} years")
    if y_limit:
        ax.set_ylim(0, np.quantile(y_limit, q=0.9))
    plt.show()


def plot_minimum_multiples(results_df: pd.DataFrame):
    min_multiples = (
        results_df.groupby("leverage")
        .apply(lambda g: g.multiple.min())
        .to_frame()
        .reset_index()
        .rename({0: "min_multiple"}, axis=1)
    )
    sns.barplot(x="leverage", y="min_multiple", data=min_multiples)
    plt.title("Minimum return multiple per leverage")
    plt.show()


def plot_fraction_of_outcomes_worse_than_reference(results: defaultdict[int, list]):
    reference = np.array(results[1])
    fract_below_ref = [
        (k, np.mean(np.array(v) < reference)) for k, v in results.items() if k != 1
    ]
    sns.barplot(
        x="leverage",
        y="fraction_of_outcomes",
        data=pd.DataFrame(
            fract_below_ref, columns=["leverage", "fraction_of_outcomes"]
        ),
    )
    plt.title("Fraction outcomes worse than non-leveraged")
    plt.show()


def plot_example_period(df: pd.DataFrame):
    df = get_random_time_span(df)
    results = dict()
    for leverage in PARAMS.leverage_values:
        results[str(leverage)] = calculate_returns(df, leverage=leverage)

    data = pd.DataFrame(results).melt(var_name="leverage", value_name="capital")
    data["time"] = df.Date.tolist() * len(results)

    sns.lineplot(data=data, x="time", y="capital", hue="leverage")
    plt.xlabel("")
    plt.title(f"Example period: {data.time.min().date()} - {data.time.max().date()}")
    plt.show()
