import pandas as pd
import collections
import six
from IPython.display import display
import numpy as np
from typing import Union, List
from copy import deepcopy


def iterable(arg):
    """Checks if an item is iterable, and not a string.

    Args:
        arg (*): The variable that we wish to check if is iterable.

    Returns:
        Boolean: Is it iterable and not a string?
    """
    return isinstance(arg, collections.Iterable) and not isinstance(
        arg, six.string_types
    )


class ArgumentValueError(Exception):
    """Exception raised when the provided argument is an invalid option."""

    def __init__(self, opt_name, opt_val, valid_vals):
        self.message = (
            f"but valid values are [{', '.join([str(o) for o in valid_vals])}]"
        )
        self.opt_name = opt_name
        self.opt_val = opt_val
        self.valid_vals = valid_vals
        super().__init__(self.message)

    def __str__(self):
        return f"{self.opt_name} -> {self.opt_val}, {self.message}"


def check_arg_value(arg_name, arg_vals, valid_vals):
    """Simple verification for arguments that require a valid option.

    Args:
        arg_name (str): The name of the argument.
        arg_val (str, numeric): The supplied value of the argument
        valid_vals (list): A list of argument values that are allowed.

    Raises:
        ValueError: The supplied argument value is invalid.

    Examples:
        >>> check_arg_value('arg1', 1, [1, 2, 3, 4, 5])

        >>> check_arg_value('arg2', 6, [1, 2, 3, 4, 5])
        ArgumentValueError: arg2 -> [6], but valid values are [1, 2, 3, 4, 5]           # noqa: E501

        >>> check_arg_value('arg3', [1, 2], [1, 2, 3, 4, 5])

        >>> check_arg_value('arg4', [1, 6], [1, 2, 3, 4, 5])
        >>> ArgumentValueError: arg4 -> [1, 6], but valid values are [1, 2, 3, 4, 5]    # noqa: E501

        >>> check_arg_value('arg5', 'cat', ['cat', 'dog', 'mouse'])

        >>> check_arg_value('arg6', ['a', 'c'], ['cat', 'dog', 'mouse'])
        ArgumentValueError: arg6 -> ['a', 'c'], but valid values are ['cat', 'dog', 'mouse']  # noqa: E501

    """
    if not iterable(arg_vals):
        arg_vals = [arg_vals]

    if not all([val in valid_vals for val in arg_vals]):
        raise ArgumentValueError(arg_name, arg_vals, valid_vals)


class NReporter:
    _ROW_LABEL = "rows"
    _COL_LABELS = ["N", "𝝙"]

    def __init__(self, group_vars: List[str] = [], nan_cols: List[str] = []):
        """Constructor for NReporter object.

        Args:
            group_vars (list, optional): A list of variables to group by
            and aggregate counts. Names should be dataframe columns
            (or index levels). Defaults to [].
            nan_cols (list, options): A list of columns that will be tabulated
            to cound NaNs.

        Examples:
            NReporter(['hatch_id', 'session_id'])

        """
        if not iterable(group_vars):
            self._groupby = [group_vars]
        else:
            self._groupby = group_vars

        if not iterable(nan_cols):
            self._nancols = [nan_cols]
        else:
            self._nancols = nan_cols

        self._counts = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [[self._ROW_LABEL] + group_vars + nan_cols, self._COL_LABELS]
            ),
            index=pd.MultiIndex.from_tuples(
                [(0, "Init")], names=["Step", "Description"]
            ),
            data=0,
        )
        self._current_i = 1

    @property
    def current_i(self) -> int:
        return self._current_i

    @property
    def prev_i(self) -> int:
        return self._current_i - 1

    @property
    def _delta_columns(self) -> List[str]:
        return [c for c in self._counts.columns if c[1] == self._COL_LABELS[1]]

    def _set_total(self, desc: str, var: str, val: int) -> None:
        _irow = (self._current_i, desc)
        _icol = (var, self._COL_LABELS[0])
        self._counts.loc[_irow, _icol] = val

    def update(self, df: pd.DataFrame, description: str) -> pd.DataFrame:
        _irow = (self._current_i, description)
        self._counts.loc[_irow, :] = 0
        self._set_total(description, self._ROW_LABEL, df.shape[0])

        for v in self._groupby:
            check_arg_value("group_var", v, list(df.columns) + list(df.index.names))
            self._set_total(description, v, df.groupby(v).ngroups)

        for v in self._nancols:
            check_arg_value("nan_cols", v, list(df.columns))
            self._set_total(description, v, df[v].count())

        self._counts.loc[:, self._delta_columns] = (
            self._counts.xs(self._COL_LABELS[0], axis=1, level=1).diff(axis=0).values
        )

        self._current_i += 1
        return df

    def apply_query(
        self, df: pd.DataFrame, query_str: str, description: Union[None, str] = None
    ) -> pd.DataFrame:
        if description is None:
            description = query_str
        return df.query(query_str).pipe(self.update, description)

    def apply_mask(
        self, df: pd.DataFrame, expr: str, mask_columns: str = "all"
    ) -> pd.DataFrame:
        _mask_columns = mask_columns
        valid_columms = list(df.columns) + ["*", "all"]

        if mask_columns in ["*", "all"]:
            mask_columns = list(df.columns)

        if not iterable(mask_columns):
            mask_columns = [mask_columns]

        for c in mask_columns:
            check_arg_value("mask_columns", c, valid_columms)

        valid_rows = df.query(expr).index
        invalid_rows = df.index.difference(valid_rows)

        df.loc[invalid_rows, mask_columns] = np.nan
        self.update(df, f"{expr} (applied to {_mask_columns})")

        return df

    def __deepcopy__(self):
        return deepcopy(self)

    def report(self):
        display(
            self._counts.iloc[1:, :]
            .droplevel(level=0, axis=0)
            .astype(int)
            .style.format({c: "{:+d}" for c in self._delta_columns})
            .set_table_styles(
                [
                    {"selector": "th", "props": [("text-align", "right")]},
                ],
                overwrite=False,
            )
        )
