"""Preprocess raw ratio/score CSVs into a per-ID time-series dictionary.

Rows that lack a matching ``(id, time)`` in either table are dropped by the
inner join. Each ID is then stored as aligned arrays
``(times, ratios, scores)`` sorted by time.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_arg() -> argparse.Namespace:
    """Parse input/output paths and the merge rule for preprocessing.

    :returns: Load/save paths, filenames, and the pandas merge method.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Merge ratio and score tables into a per-ID dictionary.")

    parser.add_argument("-l", "--load_dir",      required=False, type=str, default="../Data/Original",
                        help="Directory that contains the raw ratio and score CSVs.")
    parser.add_argument("-s", "--save_dir",      required=False, type=str, default="../Data/Preprocessed",
                        help="Directory in which to write the preprocessed dictionary.")
    parser.add_argument("-n", "--save_filename", required=False, type=str, default="id_data_dict.npy",
                        help="Filename of the saved id dictionary.")
    parser.add_argument("--ratio_file",          required=False, type=str, default="ratio_data.csv",
                        help="Filename of the ratio table inside --load_dir.")
    parser.add_argument("--score_file",          required=False, type=str, default="score_data.csv",
                        help="Filename of the score table inside --load_dir.")
    parser.add_argument("--merge_how",           required=False, type=str, default="inner",
                        choices=["inner", "left", "right", "outer"],
                        help="Join type used when merging on (id, time).")

    return parser.parse_args()


def build_id_data_dict(ratio_data, score_data, merge_how="inner"):
    """Inner-join ratio and score tables, then group them by ID.

    Rows are aligned on ``(id, time)`` and each ID is sorted by time.

    :param ratio_data: Table with columns ``id``, ``time``, ``ratio``.
    :type ratio_data: pandas.DataFrame
    :param score_data: Table with columns ``id``, ``time``, ``score``.
    :type score_data: pandas.DataFrame
    :param merge_how: Pandas merge method. Defaults to ``"inner"``, matching
        ``--merge_how``.
    :type merge_how: str
    :returns: ``{id: (times, ratios, scores)}``.
    :rtype: dict
    """
    merged = pd.merge(ratio_data, score_data, on=["id", "time"], how=merge_how)
    id_data_dict = {}
    for id_, grp in merged.groupby("id"):
        grp_sorted = grp.sort_values("time")
        id_data_dict[id_] = (
            grp_sorted["time"].values,
            grp_sorted["ratio"].values,
            grp_sorted["score"].values,
        )
    return id_data_dict


def main():
    """Merge ratio and score tables, group by ID, and write ``id_data_dict.npy``."""
    args = parse_arg()

    load_dir = Path(args.load_dir)
    ratio_data = pd.read_csv(load_dir.joinpath(args.ratio_file))
    score_data = pd.read_csv(load_dir.joinpath(args.score_file))
    id_data_dict = build_id_data_dict(ratio_data, score_data, merge_how=args.merge_how)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir.joinpath(args.save_filename)
    np.save(save_file, id_data_dict)
    print(f"Save to {save_file} .")


if __name__ == "__main__":
    main()
