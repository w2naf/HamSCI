#!/usr/bin/env python
#This code is used to generate the RBN-GOES map for the Space Weather feature article.

# import json

import numpy as np
from pandas import read_csv

from hamsci.rbn_lib import Re as earth_radius
from hamsci.rbn_lib import RbnObject
from hamsci.geopack import greatCircleDist as great_circle_dist
from datetime import timedelta

from os import makedirs
from os.path import join, exists
from json import dumps
from argparse import ArgumentParser
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument("csv_file", help="The CSV file to divide up into bins")
parser.add_argument("output_dir", help="Where to store the output data", default=".")

def load_csv_file(csv_name):

    replacement_names   = ['dx','callsign','dx_lat','dx_lon','de_lat','de_lon','freq','db','date']

    #  Load all data into a dataframe.
    df                  = read_csv(csv_name, parse_dates=[8], header=1, names=replacement_names)

    #  Calculate Total Great Circle Path Distance
    lat1, lon1          = df['de_lat'], df['de_lon']
    lat2, lon2          = df['dx_lat'], df['dx_lon']
    df.loc[:,'R_gc']    = earth_radius * great_circle_dist(lat2, lon1, lat2, lon2)

    # Calculate Band
    df.loc[:,'band']    = np.array((np.floor(df['freq'] / 1000.)), dtype=np.int)

    return df

def create_rbn_obj(recorded_data):

    rbn_obj     = RbnObject(df=recorded_data)

    rbn_obj.active.dropna()
    rbn_obj.active.filter_pathlength(500)
    rbn_obj.active.calc_reflection_points('miller2015')
    rbn_obj.active.grid_data(4)

    rbn_obj.active.compute_grid_stats()

    return rbn_obj

def save_frame(folder, name, json_string):
    if not exists(folder):
        makedirs(folder)
    file_path = join(folder, name)
    print("Saving {} into {}".format(name, file_path))
    with open(file_path, "w") as output:
        output.write(json_string + "\n")

def dump_files(rbn_obj, bin_start, bin_end, output_dir="."):
    bin_folder = join(output_dir, bin_start.strftime("%Y_%m_%d-%H_%M_%S") + bin_end.strftime("-%H_%M_%S"))

    #  Store RBN Paths and colors
    ds  = rbn_obj.DS002_pathlength_filter
    df  = ds.df
    df["color"] = ds.get_band_color(encoding="hex")
    save_frame(bin_folder, "rbn_path.json", df.to_json())

    #  Store RBN Reflection data calculations
    ds          = rbn_obj.active
    df          = ds.df
    df.index    = list(range(df.index.size))
    df["color"] = ds.get_band_color(encoding="hex")
    save_frame(bin_folder, "rbn_reflect.json", df.T.to_json())

    #  Save RBN GRid data
    df = rbn_obj.active.grid_data
    df["color"] = rbn_obj.active.get_grid_data_color(encoding="hex")
    save_frame(bin_folder, "rbn_grid.json", df.T.to_json())

    #  Save metadata about the bin. Start, stop, and integration time
    save_frame(bin_folder, "metadata.json", dumps({
        "integration_time": '15 min',
        "sTime": bin_start.strftime('%Y %b %m %H:%M UT'),
        "eTime": bin_end.strftime('%Y %b %m %H:%M UT')
    }))


def binrange(start, end, step=timedelta(minutes=15)):
    while start <= end:
        yield start, start + step
        start += step

def create_historic_bin(args):
    frame, output_dir, bin_start, bin_end = args
    frame = frame.loc[frame["date"] >= bin_start]
    frame = frame.loc[frame["date"] <= bin_end]
    dump_files(create_rbn_obj(frame), bin_start, bin_end, output_dir=output_dir)

if __name__ == '__main__':

    #  Parse user command flags
    args = parser.parse_args()

    #  Load selected CSV file into CSV
    frame = load_csv_file(args.csv_file)

    #  The time range we will use
    end   = frame["date"].max()
    start = frame["date"].min()
    times = [(frame, args.output_dir, start, end) for start, end in binrange(start, end)]

    with Pool(4) as p:
        p.map(create_historic_bin, times)
        p.join()
#    for time in times:
#        create_historic_bin(time)
