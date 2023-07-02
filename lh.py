#!/usr/bin/env python
# vim:fileencoding=utf8
#
# Project: Implementation of the Lemke-Howson algorithm for finding MNE
# Author:  Petr Zemek <s3rvac@gmail.com>, 2009
#

"""Runs a program which computes MNE in the given 2-player game
using the Lemke-Howson algorithm.
"""
import sys

sys.path.insert(0, './src')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from matplotlib import patches

from src.io import *
from src.lh import *


def visualizationEquilibrium():
    first_player = [6, 11, 1, 9]
    second_player = [6, 1, 11, 9]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.plot([1, 6], [11, 6], 'red', label='2. Játékos - 1. stratégia')
    plt.plot([6, 11], [6, 1], 'orange', label='1. Játékos - 1. stratégia')
    plt.plot([1, 9], [11, 9], 'blue', label='1. Játékos - 2. stratégia')
    plt.plot([11, 9], [1, 9], 'green', label='2. Játékos - 1. stratégia ')
    plt.plot(first_player, second_player, 'o',label='kifizetés-párok')
    plt.xlim(min(first_player) - 1, max(first_player) + 1)
    plt.ylim(min(second_player) - 1, max(second_player) + 1)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title('Kifizetési poligon')
    plt.xlabel('Első játékos kifizetései')
    plt.ylabel('Második játékos kifizetései')
    x = sy.Symbol('x')
    dn = sy.diff(n(x), x)
    soln = sy.solve(dn)
    plt.plot(soln, 12 - soln[0], 'o', markersize=8, color='brown', label='Nash egyensúlypont')
    plt.legend()
    plt.grid()
    plt.show()

def n(x):
    return 12 * x - x ** 2


def rightTriangle():
    first_player = [6, 11, 1, 9]
    second_player = [6, 1, 11, 9]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.plot(first_player, second_player, 'o')
    plt.xlim(min(first_player) - 1, max(first_player) + 1)
    plt.ylim(min(second_player) - 1, max(second_player) + 1)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('Első játékos kifizetései')
    plt.ylabel('Második játékos kifizetései')
    plt.plot([1, 6], [11, 6], 'red')
    plt.plot([6, 11], [6, 1], 'red')
    plt.plot([1,1],[11,1],color='black',label='Befogó1')
    plt.plot([11,1],[1,1],color='black',label='Befogó2')
    plt.plot(1, 1, 'o', markersize=6, color='grey', label='Valószínűségi eloszlás')
    plt.legend()
    plt.grid()
    plt.title('Nash egyensúlypont és derékszögű háromszög közti kapcsolat')
    plt.show()






def visualizationCorrelation():
    first_player = [6, 11, 1, 9]
    second_player = [6, 1, 11, 9]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    plt.plot([1, 6], [11, 6], 'red')
    plt.plot([6, 11], [6, 1], 'orange')
    plt.plot([1, 9], [11, 9], 'blue')
    plt.plot([11, 9], [1, 9], 'green')
    plt.plot(first_player, second_player, 'o')
    plt.xlim(min(first_player) - 1, max(first_player) + 1)
    plt.ylim(min(second_player) - 1, max(second_player) + 1)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('Első játékos kifizetései')
    plt.ylabel('Második játékos kifizetései')
    x = sy.Symbol('x')
    dn = sy.diff(n(x), x)
    soln = sy.solve(dn)
    plt.plot(soln, 12 - soln[0], 'o', markersize=8, color='brown', label='Nash egyensúlypont')
    plt.plot(1,1,'o' ,markersize=6,color='grey', label='Valószínűségi eloszlás')
    plt.plot(1/6,1/6, 'o',markersize=7, color='purple', label='Nem normalizált valószínűségi eloszlás')
    plt.plot([1/6,6],[1/6,6],color='black',label='Relációs egyenes')
    plt.legend()
    plt.grid()
    plt.title('A 3 pont közötti összefüggés')
    plt.show()


def main():
    try:
        # These imports must be here because of possible
        # SyntaxError exceptions in different versions of python
        # (this program needs python 2.5)

        # Check program arguments (there should be none)
        if len(sys.argv) > 1:
            stream = sys.stderr
            if sys.argv[1] in ['-h', '--help']:
                stream = sys.stdout
            printHelp(stream)

            return 1

        # Obtain input matrices from the standard input
        m1, m2 = parseInputMatrices(sys.stdin.read())

        # Compute the equilibirum
        eq = lemkeHowson(m1, m2)

        # Print both matrices and the result
        printGameInfo(m1, m2, eq, sys.stdout)
        sys.stdout.write("\n")

        uneq = unnormalizedEquilibrium(m1, m2)
        sys.stdout.write("Unnormalized equilibrium:\n")
        print(uneq)

        # Visualizations
        visualizationEquilibrium()
        visualizationCorrelation()
        rightTriangle()

        return 0
    except SyntaxError:
        sys.stderr.write('Need python 2.5 to run this program.\n')


if __name__ == '__main__':
    main()
