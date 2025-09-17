"""
Description: Generic helper functions I use in projects. 

Sections:
- Data viz
- Stats
- Input/output

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
import statistics as st
from scipy import stats
from scipy.stats import bootstrap
import logging
import json 
from datetime import datetime


# DATA VIZ
###################################
###################################

def make_aesthetic(hex_color_list=None, 
	with_gridlines=False, 
	bold_title=False, 
	save_transparent=False, 
	font_scale=2, 
	latex2arial = True
	):
    """Make Seaborn look clean and add space between title and plot"""
    
    # Note: To make some parts of title bold and others not bold, we have to use
    # latex rendering. This should work: 
    # plt.title(r'$\mathbf{bolded\ title}$' + '\n' + 'And a non-bold subtitle')

    
    sns.set(style='white', context='paper', font_scale=font_scale)
    if not hex_color_list:
        hex_color_list = [
            "#2C3531",  # Dark charcoal gray 
            "#F45B69",  # Vibrant pinkish-red (red)
            "#00A896",  # Persian green (green)
            "#E3B505",  # Saffron (yellow)
            "#826AED",  # Medium slate blue (purple)
            "#F18805",  # Tangerine (orange)
            "#89DAFF",  # Pale azure (cyan)
            "#7E6551",  # Coyote (brown/grey)
            "#D41876",  # Telemagenta
            "#020887",  # Phthalo blue
            "#7DCD85",  # Emerald
            "#E87461",  # Medium-bright orange
            "#342E37",  # Dark grayish-purple
            "#F7B2AD",  # Melon
            "#D4B2D8",  # Pink lavender
        ]
    
    sns.set_palette(sns.color_palette(hex_color_list))

    # Update on 
    # 2024-11-29: I realized I can automatically 
    # clean variable names so i dont have to manually replace underscore
    
    # Enhanced typography settings
    plt.rcParams.update({
        # font settings
        'font.family': 'Arial',
        'font.weight': 'regular',
        'axes.labelsize': 12 * font_scale,
        'axes.titlesize': 16 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,
        
        # spines/grids
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,  # Thinner spines
        'axes.grid': with_gridlines,
        'grid.alpha': 0.2,       
        'grid.linestyle': ':', 
        'grid.linewidth': 0.5,
        
        # title
        'axes.titlelocation': 'left',
        'axes.titleweight': 'bold' if bold_title else 'regular',
        'axes.titlepad': 15 * (font_scale / 1),
        
        # fig
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.2,
        'figure.constrained_layout.w_pad': 0.2,
        
        # legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.borderpad': 0.4,
        'legend.borderaxespad': 1.0,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.7,
        'legend.handletextpad': 0.5,
        
        # export
        'savefig.dpi': 300,
        'savefig.transparent': save_transparent,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'figure.autolayout': False,
        
         # do this for the bold hack
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'

    })
    
    return hex_color_list


def smart_legend(ax=None, position='auto', **legend_kwargs):
    """Intelligent legend positioning that prevents overlap"""
    if ax is None:
        ax = plt.gca()

    legend_defaults = {
        'frameon': True,
        'framealpha': 0.95,
        'facecolor': 'white',
        'edgecolor': '#CCCCCC',
    }
    legend_defaults.update(legend_kwargs)

    if position == 'outside_right':
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', **legend_defaults)
    elif position == 'outside_bottom':
        legend = ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
                           ncol=legend_kwargs.get('ncol', 3), **legend_defaults)
    else:
        legend = ax.legend(loc='best', **legend_defaults)

    return legend

def clean_vars(s, how='title'):
    """
    Simple function to clean titles

    Params
    - s: The string to clean
    - how (default='title'): How to return string. Can be either ['title', 'lowercase', 'uppercase']

    Returns
    - cleaned string
    """
    assert how in ['title', 'lowercase', 'uppercase'], "Bad option!! see docs"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    s = s.replace('_', ' ')
    if how == 'title':
        return s.title()
    elif how=='lower':
        return s.lower()
    elif how=='upper':
    	return s.upper()

def make_bold(x):
    """
    This function is used to make part of the title bold like as a subtitle.
    Basically, it's using latex to render a (sub)string bold in matplotlib.
    But make_aesthetic() should handle using Arial for math font, so it won't look weird.
    
    >>> full_title = f"{make_bold('Regression Coefficients of Estimated Prevalence From Multiverse')}\\n(Baseline is raw data with no weighting and no dropping)"
    """
    words = x.split()
    words = r'\ '.join([w for w in words])  # Escape backslash properly
    bold_str = f"$\\bf{{{words}}}$"  # Correctly format the f-string
    return bold_str

###################################
###################################


# STATS
###################################
###################################

def cat_stats(data, include_n=True, digits=1, sort_by='frequency', reverse=True):
    """Calculate and format statistics for categorical data.

    This function analyzes categorical data by computing frequencies and percentages
    for each unique category. It returns both a formatted string for display and
    a dict. You can choose how you want to order the categories. Its the cat version of arraystats.

    Args:
        data (array-like): The input categorical data array. Can be a list,
            numpy array, or any iterable containing categorical values.
        include_n (bool, optional): Whether to include raw counts (n=X) in the
            formatted output string. Defaults to True.
        digits (int, optional): Number of decimal places for percentage rounding
            in the output. Defaults to 1.
        sort_by (str, optional): How to sort the results. Options are:
            - 'frequency': Sort by frequency/count (default)
            - 'alphabetical': Sort alphabetically by category name
            - 'original': Maintain order of first appearance in data
        reverse (bool, optional): Whether to reverse the sort order. Defaults to
            True for descending frequency (most common first).

    Returns:
        tuple: A tuple containing two elements:
            - str: Formatted string showing categories with percentages and
              optionally counts in the format "Category1 (45.2%; n=10), 
            - dict: Dictionary with categories as keys and dictionaries as values
              containing:
              - 'count': Raw frequency count
              - 'percentage': Exact percentage (float)  
              - 'percentage_rounded': Rounded percentage for display

    """
    # Convert to numpy array for consistency
    data = np.array(data)
    # Count frequencies
    counter = Counter(data)
    total_count = len(data)

    # Calculate percentages and build results dictionary
    result_dict = {}
    for category, count in counter.items():
        percentage = (count / total_count) * 100
        result_dict[category] = {
            'count': count,
            'percentage': percentage,
            'percentage_rounded': round(percentage, digits)
        }

    # Sort results according to preference
    if sort_by == 'frequency':
        sorted_items = sorted(result_dict.items(),
                              key=lambda x: x[1]['count'],
                              reverse=reverse)
    elif sort_by == 'alphabetical':
        sorted_items = sorted(result_dict.items(),
                              key=lambda x: str(x[0]),
                              reverse=reverse)
    elif sort_by == 'original':
        # Maintain order of first appearance in the data
        seen = set()
        original_order = []
        for item in data:
            if item not in seen:
                seen.add(item)
                original_order.append(item)
        sorted_items = [(item, result_dict[item]) for item in original_order]
    else:
        raise ValueError("sort_by must be 'frequency', 'alphabetical', or 'original'")

    # Format output string
    result_parts = []
    for category, stats in sorted_items:
        if include_n:
            part = f"{category} ({stats['percentage_rounded']:.{digits}f}%; n={stats['count']})"
        else:
            part = f"{category} ({stats['percentage_rounded']:.{digits}f}%)"
        result_parts.append(part)

    result_string = ", ".join(result_parts)

    # Return both the formatted string and the full dictionary for programmatic use
    return result_string, {k: v for k, v in sorted_items}


def array_stats(data, digits=2, include_ci=False):
    """Calculate and print summary statistics for an array.

    This function computes basic descriptive statistics (mean, median, standard
    deviation, and mode) for a given array of data. It also provides an option
    to calculate and include a 95% confidence interval for the mean using
    bootstrap resampling.

    Args:
        data (array-like): The input data array for which statistics will be
            calculated. Can be a list, numpy array, or any array-like structure.
        digits (int, optional): Number of decimal places for rounding the
            calculated statistics. Defaults to 2.
        include_ci (bool, optional): Whether to include a 95% confidence
            interval for the mean using bootstrap resampling. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics with the
            following keys:
            - 'mean': The arithmetic mean of the data
            - 'median': The median (middle value) of the data  
            - 'sd': The sample standard deviation (using ddof=1)
            - 'mode': The most frequently occurring value
            - 'ci': (optional) A tuple containing the lower and upper bounds
              of the 95% confidence interval, only present if include_ci=True


    Note:
        The function prints formatted statistics to the console like this--
        "M = 2.83, SD = 1.47, Mdn = 2.50" \n "Mode = 2.00" in addition to
        returning them as a dictionary. If include_ci=True, it also prints the
        confidence interval as "95% CI [1.83, 3.83]". When multiple modes exist,
        the function will print a warning message and use the first occurrence.
        
        The bootstrap uses Scipy 10K iterations and is bootstrapping the mean. 
    """
    data = np.array(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    sd_val = np.std(data, ddof=1)

    # Calculate mode - handling cases with multiple modes
    try:
        mode_val = st.mode(data)
    except st.StatisticsError:
        # If multiple modes, use scipy's mode which returns the first occurrence
        print("Multiple modes found, using the first one.")
        mode_val = stats.mode(data, keepdims=True)[0][0]

    result = {
        'mean': round(mean_val, digits),
        'median': round(median_val, digits),
        'sd': round(sd_val, digits),
        'mode': round(mode_val, digits)
    }

    # Add confidence interval if requested
    if include_ci:
        def mean_func(x, axis):
            return np.mean(x, axis=axis)

        data_reshaped = np.array(data).reshape(-1, 1)
        bootstrap_result = bootstrap((data_reshaped,), mean_func,
                                     confidence_level=0.95,
                                     random_state=42,
                                     n_resamples=10 * 1000)
        ci_lower, ci_upper = bootstrap_result.confidence_interval
        result['ci'] = (round(float(ci_lower), digits), round(float(ci_upper), digits))

    print(f"M = {result['mean']:.{digits}f}, SD = {result['sd']:.{digits}f}, Mdn = {result['median']:.{digits}f}")
    print(f"Mode = {result['mode']:.{digits}f}")
    if include_ci and 'ci' in result:
        print(f"95% CI [{result['ci'][0]:.{digits}f}, {result['ci'][1]:.{digits}f}]")

    return result


def format_p_value(p, threshold=0.001, exact=True, digits=2):
    """
    Format p-values consistently for publication and presentation.
    
    This function standardizes p-value formatting following APA guidelines,
    with options for exact values or thresholding for very small p-values.
    
    Args:
        p (float): The p-value to format
        threshold (float, optional): Threshold below which to report as "p < threshold".
            Defaults to 0.001.
        exact (bool, optional): If True, always report exact p-value regardless of
            threshold. Defaults to False.
        digits (int, optional): Number of decimal places for exact p-values.
            Defaults to 3.
    
    Returns:
        str: Formatted p-value string (e.g., "p < .001", "p = .043", "p = .200")
    
    """
    # Handle edge cases
    if p < 0 or p > 1:
        raise ValueError("p-value must be between 0 and 1")

    # If exact=True, use scientific notation for values below threshold
    if exact:
        if p < threshold:
            sci_digits = 0 if p == float(f"{p:.0e}") else 1
            return f"p = {p:.{sci_digits}e}"
        else:
            return f"p = {p:.{digits}f}".replace("0.", ".")

    # If not exact and p below threshold, report as "p < threshold"
    else:
        if p < threshold:
            return f"p < {threshold}".replace("0.", ".")

    # But for values at or above threshold, report exact value
    formatted_p = f"{p:.{digits}f}".replace("0.", ".")
    return f"p = {formatted_p}"


def extreme_by_col(df, col, n_per_extreme=10):
   """Get top and bottom N rows sorted by column"""
   sorted_df = df.sort_values(col, ascending=False)
   
   top_n = sorted_df.head(n_per_extreme).copy()
   top_n['extreme'] = 'top'
   
   bottom_n = sorted_df.tail(n_per_extreme).copy()
   bottom_n['extreme'] = 'bottom'
   
   return pd.concat([top_n, bottom_n], ignore_index=True)
	
###################################
###################################

# I/O UTILS
###################################
###################################
def list2text(data_list, filename):
    """Write list to text file, one item per line"""
    with open(filename, 'w') as f:
        f.write('\n'.join(map(str, data_list)))

def text2list(filename):
    """Read text file into list, one line per item"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]
	    
def timestamp(style='readable'):
    """Return current timestamp string for filenames"""
    if style == 'readable':
        return datetime.now().strftime('%Y-%m-%d__%H.%M.%S')
    elif style == 'compact':
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    elif style == 'date_only':
        return datetime.now().strftime('%Y-%m-%d')

def dict2json(data_dict, filename):
    """Save dictionary to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=2)

def json2dict(filename):
    """Load dictionary from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def log_and_print(message):
    """Log a message using the current logger and print it to console."""
    logger = logging.getLogger()
    logger.info(message)
    print(message)
###################################
###################################
