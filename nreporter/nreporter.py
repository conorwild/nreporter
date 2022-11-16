import pandas as pd
import collections
import six
from IPython.display import display

def iterable(arg):
    """ Checks if an item is iterable, and not a string.

    Args:
        arg (*): The variable that we wish to check if is iterable.

    Returns:
        Boolean: Is it iterable and not a string?
    """
    return (
        isinstance(arg, collections.Iterable)
        and not isinstance(arg, six.string_types)
    )

class ArgumentValueError(Exception):
    """ Exception raised when the provided argument is an invalid option.
    """
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
    """ Simple verification for arguments that require a valid option.

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


class NReporter():

    _ROW_LABEL = 'rows'
    _COL_LABELS = ['N', u'𝝙']

    def __init__(self, group_vars=[], nan_cols=[]):
        """ Constructor for NReporter object.

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
                [[self._ROW_LABEL]+group_vars+nan_cols, self._COL_LABELS]
            ),
            index=pd.MultiIndex.from_tuples(
                [(0, 'Init')], names=['Step', 'Description']
            ),
            data=0
        )
        self._current_i = 1

    @property
    def current_i(self):
        return self._current_i

    @property
    def prev_i(self):
        return self._current_i-1

    @property
    def _delta_columns(self):
        return [
            c for c in self._counts.columns if c[1] == self._COL_LABELS[1]
        ]

    def _set_total(self, desc, var, val):
        _irow = (self._current_i, desc)
        _icol = (var, self._COL_LABELS[0])
        self._counts.loc[_irow, _icol] = val

    def update(self, df, description):
        _irow = (self._current_i, description)
        self._counts.loc[_irow, :] = 0
        self._set_total(description, self._ROW_LABEL, df.shape[0])

        for v in self._groupby:
            check_arg_value(
                'group_var', v, list(df.columns)+list(df.index.names)
            )
            self._set_total(description, v, df.groupby(v).ngroups)

        for v in self._nancols:
            check_arg_value(
                'nan_cols', v, list(df.columns)
            )
            self._set_total(description, v, df[v].count())

        self._counts.loc[:, self._delta_columns] = (
            self._counts
            .xs(self._COL_LABELS[0], axis=1, level=1)
            .diff(axis=0)
            .values
        )

        self._current_i += 1
        return df

    def __deepcopy__(self):
        return deepcopy(self)

    def report(self):
        display(
            self._counts
            .iloc[1:, :]
            .droplevel(level=0, axis=0)
            .astype(int)
            .style
            .format(
                {c: '{:+d}' for c in self._delta_columns}
            )
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'right')]},
            ], overwrite=False)
        )
