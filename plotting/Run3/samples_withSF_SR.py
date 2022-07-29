import os
import ROOT
from auxiliars import *

def hdf5inpath(path):
  ret = []
  for f in os.listdir(path):
    if "hdf5" in f: 
      ret.append(path + "/" + f)
  return ret

order   = ["ttZ","VV", "WJets","ttto1l", "ttto2l", "DY_Pt650ToInf", "DY_Pt400To650","DY_Pt250To400","DY_Pt100To250","DY_Pt50To100", "DY_Pt0To50"]
samples = {
  "data": {
         "name" : "data",
         "label": "Data",
         "xsec" : -1,
         "lineColor": ROOT.kBlack,
         "fillcolor": ROOT.kBlack,
         "isSig"    : False,
         "files"    : hdf5inpath("/eos/user/c/cericeci/SUEP/28_05_2022//data/"),#+ hdf5inpath("/eos/user/c/cericeci/SUEP/25_07_2022_SRonly/data_RunB/")+hdf5inpath("/eos/user/c/cericeci/SUEP/25_07_2022_SRonly/data_RunC/")+hdf5inpath("/eos/user/c/cericeci/SUEP/25_07_2022_SRonly/data_RunD/"),
         "markerstyle": 20,
         "markersize" : 1,
  },
  "DY": {
         "name"       : "DY", #Here plain text
         "label"      : "DY", #Here we can use weird glyphs
         "xsec"       : 6480*1000., # in fb
         "linecolor"  : ROOT.kBlack,
         "fillcolor"  : 7, # White
         "isSig"      : False,
         "extraWeights": lambda x: 1, 
         "files"      : hdf5inpath("/eos/user/c/cericeci/SUEP/28_05_2022/DY/"),
  },      
  "ttto2l": {
         "name"     : "ttto2l", #Here plain text
         "label"    : "t#bar{t} (2l)", #Here we can use weird glyphs
         "xsec"     : 922.5*((3*0.108)**2)*1000., # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 2, # Red
         "isSig"    : False,
         "files"    :  hdf5inpath("/eos/user/c/cericeci/SUEP/28_05_2022/TTTo2L2Nu/"),
         "extraWeights": lambda x: 1,
  },
  "ttto1l": {
         "name"     : "ttto1l", #Here plain text
         "label"    : "t#bar{t} (1l)", #Here we can use weird glyphs
         "xsec"     : 922.5*(3*0.108)*(1-3*0.108)*1000., # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 5, # Yellow
         "isSig"    : False,
         "files"    : hdf5inpath("/eos/user/c/cericeci/SUEP/28_05_2022/TTTo2J1L1Nu/"),
         "extraWeights": lambda x: 1,
  },
  "Wjets": {
         "name"     : "Wjets", #Here plain text
         "label"    : "W", #Here we can use weird glyphs
         "xsec"     : 22808.9*1000, # in fb
         "linecolor": ROOT.kBlack,
         "fillcolor": 6, # Purple
         "isSig"    : False,
         "files"    : hdf5inpath("/eos/user/c/cericeci/SUEP/28_05_2022/WJets/"), 
         "extraWeights": lambda x: 1,
  },
}
