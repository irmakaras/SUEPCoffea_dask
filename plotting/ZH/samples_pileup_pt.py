import os
import ROOT
from auxiliars import *
import numpy as np
import pandas as pd

def hdf5inpath(path):
  ret = []
  for f in os.listdir(path):
    if "hdf5" in f: 
      ret.append(path + "/" + f)
  return ret


# Main path where samples are stored
samples = {

  
  "Pileup0t10": {
         "name"     : "Pileup0to10", #Here plain text
         "label"    : "0 #leq PU < 10", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kYellow,
         "fillcolor": ROOT.kYellow,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 0)*(x["injected_PU"] < 10)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
      "Pileup10t20": {
         "name"     : "Pileup10to20", #Here plain text
         "label"    : "10 #leq PU < 20", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kRed,
         "fillcolor": ROOT.kRed,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 10)*(x["injected_PU"] < 20)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
  "Pileup20t30": {
         "name"     : "Pileup20to30", #Here plain text
         "label"    : "20 #leq PU < 30", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kGreen,
         "fillcolor": ROOT.kGreen,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 20)*(x["injected_PU"] < 30)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
  "Pileup30t40": {
         "name"     : "Pileup30to40", #Here plain text
         "label"    : "30 #leq PU < 40", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kBlue,
         "fillcolor": ROOT.kBlue,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 30)*(x["injected_PU"] < 40)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
  "Pileup40t60": {
         "name"     : "Pileup40", #Here plain text
         "label"    : "40 #leq PU < 60", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kMagenta,
         "fillcolor": ROOT.kMagenta,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 40)*(x["injected_PU"] < 60)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
  "Pileup60Up": {
         "name"     : "Pileup60Up", #Here plain text
         "label"    : "PU #geq 60", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": ROOT.kBlack,
         "isSig"    : True, 
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: (x["injected_PU"] >= 60)*x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },

  "Inclusive": {
         "name"     : "Inclusive", #Here plain text
         "label"    : "Inclusive", #Here we can use weird glyphs
         "xsec"     : 870 * 0.0336 * 2, # in fb
         "linecolor": ROOT.kGray,
         "fillcolor": ROOT.kGray,
         "isSig"    : False, # So this is a histogram
         "files"    : hdf5inpath("/eos/home-i/iaras/SUEP/suep-data/run3-400k"),
         "extraWeights": lambda x: x["PUWeight"]*x["L1prefireWeight"]*x["bTagWeight"]*x["TrigSF"],
  },
}