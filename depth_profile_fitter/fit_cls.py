from collections import namedtuple
import numpy as np
import pandas as pd

# Define the named tuple
NumberSpec = namedtuple('NumberSpec', ['value', 'key'])

class RangeHelper:
    """
    Helper class for creating parameter ranges and translating between index values and data values.
    """
    
    def __init__(self, data, **kwargs):
        """
        Initialize the RangeHelper with data.
        
        Parameters:
        -----------
        data : pd.DataFrame or dict
            The dataset to work with.
        """
        self.data = self._ensure_dataframe(data)

    
    def _ensure_dataframe(self, data):
        """Ensure the data is in a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be a pandas DataFrame or a dictionary.")
    
    def _parse_value(self, value, column):
        """Parse a value, translating 'max' and 'range' into numerical values."""
        if isinstance(value, str):
            if value.lower() == "max":
                return self.data[column].max()
            elif value.lower() == "range":
                return self.data[column].max() - self.data[column].min()
            else:
                raise ValueError(f"Unknown string value: {value}")
        return value
    
    def create_range(self, param_info, size):
        """
        Create a parameter range with the specified scale.
        
        Parameters:
        -----------
        param_info : list
            [name, min, max, scale] for the parameter.
        size : int
            Number of points in the range.
        
        Returns:
        --------
        np.ndarray
            The created range.
        """
        name, min_val, max_val, scale = param_info
        min_val = self._parse_value(min_val, name)
        max_val = self._parse_value(max_val, name)
        
        if scale.lower() == "linear":
            return np.linspace(min_val, max_val, size)
        elif scale.lower() == "log":
            return np.logspace(np.log10(min_val) if min_val > 0 else -20, 
                               np.log10(max_val), size)
        elif scale.lower() == "index":
            return np.array(range(min_val, max_val + 1))[:size]
        else:
            raise ValueError(f"Unknown scale type: {scale}")
    
    def get_index(self, value, column):
        """
        Get the index value for a data value.
        
        Parameters:
        -----------
        value : float
            The data value.
        column : str
            The column name.
        
        Returns:
        --------
        int
            The index value.
        """
        return self.data[self.data[column] == value].index[0]
    
    def get_data(self, index, column):
        """
        Get the data value for an index value.
        
        Parameters:
        -----------
        index : int
            The index value.
        column : str
            The column name.
        
        Returns:
        --------
        float
            The data value.
        """
        return self.data.loc[index, column]


class MatrixOps:
    """Higher level operator"""

    _type = "matrix_operator"

    def __init__(
        self,
        data,
        xrange=[None, None, None, None],
        yrange=[None, None, None, None],
        min_range=2,
        size=50,
        **kwargs
    ):
        self.data = data
        self.size = size
        # self.std_values = BaseProfile(sims_obj).std_values
        self.max_ind = self.data.index.max()
        self.max_depth = self.data["Depth"].max()

        # if "fit" in cls_type.lower() and xrange[0] is None:
        #     xrange = ["depth", None, None, "index"]
        # if "fit" in cls_type.lower() and yrange[0] is None:
        #     yrange = ["depth", None, None, "index"]
        # if "pred" in cls_type.lower() and xrange[0] is None:
        #     xrange = ["conc", None, None, "log"]
        # if "pred" in cls_type.lower() and yrange[0] is None:
        #     yrange = ["diff", None, None, "log"]

        self.xrange = xrange
        self.yrange = yrange
        self.min_range = min_range

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def xrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_xrange"):
            self.xrange = ["depth", None, None, "index"]
        return self._xrange

    @xrange.setter
    def xrange(self, value):
        self._xrange = self._range_maker(value)


    @property
    def yrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_yrange"):
            self.yrange = ["depth", None, None, "index"]
        return self._yrange

    @yrange.setter
    def yrange(self, value):
        self._yrange = self._range_maker(value)


    def _range_maker(self, name, start, stop, size, abs_max):
        # if start is None:
        #     start = 0
        # if stop is None:
        #     stop = 100
        ind_max = len(self.data)
        data_max = self.data[name].max() if name in self.data.columns else self.data.iloc[:, 0].max()
        if isinstance(abs_max, int) and abs_max < len(self.data):
            ind_max = len(self.data)
            data_max = self.data[ind_max, name] if name in self.data.columns else self.data.iloc[ind_max, 0]
        elif isinstance(abs_max, (int, float)) and abs_max > len(self.data):
            ind_max = abs_max
            data_max = abs_max
        
        if "ind" in name.lower():
            res = np.linspace(
                start,
                min(stop, ind_max),
                min(size, ind_max + 1),
                dtype=int,
            )
        elif "lin" in name.lower():
            res = np.linspace(
                start,
                min(stop, self.max_depth),
                min(size, ind_max + 1),
            )
        elif "log" in name.lower():
            res = np.logspace(
                start, stop, min(size, ind_max + 1)
            )
        else:
            res = np.array(range(min(size, ind_max + 1)))

        return res

    def matrix_generator(self, constraint = None):
        range_lim = self.xrange[self.min_range] - self.xrange[0]
        constraint = constraint or (lambda x, y: x + range_lim <= y)
        for x in self.xrange:
            for y in self.yrange:
                if constraint(x, y):
                    yield (x, y)

            # self.obj_operator = Composite()
        # if "fit" in cls_type.lower():
        #     range_lim = self.xrange[min_range] - self.xrange[0]
        #     [
        #         self.obj_operator.add(
        #             ProfileOps(
        #                 FitProfile(sims_obj, start_index=x, stop_index=y, **kwargs),
        #                 **kwargs
        #             )
        #         )
        #         for x in self.xrange
        #         for y in self.yrange
        #         if (x + range_lim <= y)
        #     ]
        # if "pred" in cls_type.lower():
        #     [
        #         self.obj_operator.add(
        #             ProfileOps(
        #                 PredProfile(sims_obj, diff=y, conc=x, **kwargs), **kwargs
        #             )
        #         )
        #         for x in self.xrange
        #         for y in self.yrange
        #     ]
        # if self.obj_operator._family[0].prof.min_range != min_range:
        #     self.obj_operator.set_attr(attr="min_range", num=min_range, limit=False)

    # def error_calc(self, get_best=True, **kwargs):
    #     """Return sum of squared errors (pred vs actual)."""
    #     if get_best:
    #         self.obj_operator.set_best_error(**kwargs)
    #     else:
    #         self.obj_operator.set_error(**kwargs)