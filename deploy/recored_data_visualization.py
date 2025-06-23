import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


def load_all_pickles_from_dir(directory):
    data_list = []
    files = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    for f in files:
        path = os.path.join(directory, f)
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
            data_list.append(data)
    return data_list


def extract_field(data_list, field_path):
    extracted = []
    for data in data_list:
        d = data
        for key in field_path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                d = None
                break
        extracted.append(d)
    return np.array(extracted)


def normalize_array(arr):
    if arr.ndim == 1:
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        else:
            return arr
    else:
        normed = []
        for d in range(arr.shape[1]):
            col = arr[:, d]
            min_val, max_val = np.min(col), np.max(col)
            if max_val - min_val > 1e-8:
                normed_col = (col - min_val) / (max_val - min_val)
            else:
                normed_col = col
            normed.append(normed_col)
        return np.stack(normed, axis=1)


def plot_multiple_fields(data_list, field_paths, dims_per_field):
    T = len(data_list)
    normalize = len(field_paths) > 1  # need normalization when there are multiple fields

    plt.figure(figsize=(10, 6))

    for i, field_path in enumerate(field_paths):
        data_array = extract_field(data_list, field_path)
        label_prefix = '/'.join(field_path)

        if isinstance(data_array[0], (float, int)):
            data_array = np.array(data_array)
            if normalize:
                data_array = normalize_array(data_array)
            plt.plot(range(T), data_array, label=f"{label_prefix}")
        else:
            data_array = np.stack(data_array)
            if normalize:
                data_array = normalize_array(data_array)
            dims = dims_per_field[i]
            for d in dims:
                plt.plot(range(T), data_array[:, d], label=f"{label_prefix}[{d}]")

    plt.xlabel("Time step")
    plt.title("Normalized" if normalize else "Raw" + " Field(s) Over Time")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def flatten_keys(d, prefix=None):
    keys = []
    for k, v in d.items():
        path = [k] if prefix is None else prefix + [k]
        if isinstance(v, dict):
            keys.extend(flatten_keys(v, path))
        else:
            keys.append(path)
    return keys


def main():
    # GUI choose directory
    # root = tk.Tk()
    # root.withdraw()
    folder = filedialog.askdirectory(title="select a data folder that includes pkl files")
    data_list = load_all_pickles_from_dir(folder)
    print(f"Loaded {len(data_list)} records.")

    # get data paths
    sample = data_list[0]
    field_paths_all = flatten_keys(sample)
    print("\nvisualized field names:")
    for idx, path in enumerate(field_paths_all):
        print(f"{idx}: {'/'.join(path)}")

    # 用户选择多个字段
    choices = input("please input the indices of the fields to visualize (multiple indices are split by a single ','):"
                   ).strip().split(',')
    selected_paths = []
    dims_per_field = []

    for choice in choices:
        try:
            field_path = field_paths_all[int(choice)]
            selected_paths.append(field_path)

            # 提取数据确定维度
            data_array = extract_field(data_list, field_path)
            if isinstance(data_array[0], (float, int)):
                dims_per_field.append([])
            else:
                data_array = np.stack(data_array)
                D = data_array.shape[1]
                dims_input = input(
                    f"the dimension of the field {'/'.join(field_path)} is {D}, please input the dimension to visualize (e.g., `0,1`) (empty input denotes visualizing all dimensions):"
                ).strip()
                dims = list(map(int, dims_input.split(','))) if dims_input else list(range(D))
                dims_per_field.append(dims)

        except:
            breakpoint()
            print(f"invalid input: {choice}")
            return

    plot_multiple_fields(data_list, selected_paths, dims_per_field)


if __name__ == "__main__":
    main()
