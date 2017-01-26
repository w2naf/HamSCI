#!/usr/bin/env python
import numpy as np
import pandas as pd
import hamsci

from hamsci.gridsquare import gridsquare_grid

class AdjacentGrids(object):
    def __init__(self,precision=4):
        self.grid       = gridsquare_grid(precision=precision)
        self.grid_flat  = self.grid.flatten()

    def __bump_inx__(self,inx,ew_bump,ns_bump):
        """
        Move the index of a grid square while wrapping around the grid.
        """
        ew  = inx[0] + ew_bump
        ns  = inx[1] + ns_bump

        ew  = ew % self.grid.shape[0]
        ns  = ns % self.grid.shape[1]

        return (ew,ns)

    def bump_grid(self,gridsquare,ew_bump,ns_bump):
        inx_flat    = np.where(self.grid_flat == gridsquare)[0][0]
        inx_2d      = np.unravel_index(inx_flat,self.grid.shape)

        new_inx     = self.__bump_inx__(inx_2d,ew_bump,ns_bump)
        new_grid    = self.grid[new_inx]
        return new_grid

    def get_adjacent(self,gridsquare):
        bump_dct = {}
        bump_dct['N']    = ( 0, 1)
        bump_dct['NE']   = ( 1, 1)
        bump_dct['E']    = ( 1, 0)
        bump_dct['SE']   = ( 1,-1)
        bump_dct['S']    = ( 0,-1)
        bump_dct['SW']   = (-1,-1)
        bump_dct['W']    = (-1, 0)
        bump_dct['NW']   = (-1, 1)

        adj_dct = {}
        good = True
        for key,bump in bump_dct.iteritems():
            res          = self.bump_grid(gridsquare,bump[0],bump[1])
            
            if (gridsquare[1] == 'R' and res[1] == 'A')   \
            or (gridsquare[1] == 'A' and res[1] == 'R'):
                good = False
                break
            adj_dct[key] = res

        if not good:
            adj_dct = {}
            for key,bump in bump_dct.iteritems():
                adj_dct[key] = 'undef'

        return adj_dct

adj_grid    = AdjacentGrids(precision=4)
grids       = adj_grid.grid_flat

adjacents   = []
for grid in grids:
    adjacents.append(adj_grid.get_adjacent(grid))

df              = pd.DataFrame(adjacents,index=grids)
df.index.name   = 'gridsquare'

dirs            = ['N','NE','E','SE','S','SW','W','NW']
df[dirs].to_csv('adjacency_matrix.csv')

import ipdb; ipdb.set_trace()
