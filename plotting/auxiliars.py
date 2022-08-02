import ROOT
import numpy as np
import os

##### SF files #####
#####  -UL18-  #####
SUEP_BASE = os.path.dirname(os.path.realpath(__file__).replace("/plotting",""))

muIDfile = ROOT.TFile(SUEP_BASE + "/data/MuUL18/MuID.root","READ")
muID = muIDfile.Get("NUM_LooseID_DEN_TrackerMuons_abseta_pt")
maxmidx = muID.GetNbinsX()
maxmidy = muID.GetNbinsY()

muTrkfile = ROOT.TFile(SUEP_BASE + "/data/MuUL18/MuTracking.root","READ")
muTrk = muTrkfile.Get("NUM_TrackerMuons_DEN_genTracks")
maxmtx = muTrk.GetNbinsX()
maxmty = muTrk.GetNbinsY()

eIDfile = ROOT.TFile(SUEP_BASE + "/data/EGammaUL18/EGamma_Loose.root","READ")
eID = eIDfile.Get("EGamma_SF2D").Clone("eID") #Clone just in case pyroot does its thing
maxeidx = eID.GetNbinsX()
maxeidy = eID.GetNbinsY()

eRECOfile = ROOT.TFile(SUEP_BASE + "/data/EGammaUL18/EGamma_recohighpt.root", "READ")
eRECO = eRECOfile.Get("EGamma_SF2D").Clone("eRECO") #Clone just in case pyroot does its thing
maxerex = eRECO.GetNbinsX()
maxerey = eRECO.GetNbinsY()


pufile = ROOT.TFile("/eos/user/c/cericeci/Run3/pileupWeights.root", "READ")
puhisto = pufile.Get("hRatio").Clone("puWeights")
maxpuid = puhisto.GetNbinsX()

def PU(x):
  w = np.array([])
  for i in range(len(x)):
    w = np.append(w,puhisto.GetBinContent(min(max(1, puhisto.GetXaxis().FindBin(x["Pileup_nTrueInt"][i])), maxpuid)))
  return w


def getSFMu(pt, eta):
  sf  = muID.GetBinContent(min(max(1,muID.GetXaxis().FindBin(abs(eta))),maxmidx), min(max(1,muID.GetYaxis().FindBin(pt)),maxmidy))
  sf *= muTrk.GetBinContent(min(max(1,muTrk.GetXaxis().FindBin(abs(eta))),maxmtx), min(max(1,muTrk.GetYaxis().FindBin(pt)),maxmty))
  return sf

def getSFEl(pt, eta):
  sf  = eID.GetBinContent(min(max(1,eID.GetXaxis().FindBin(eta)),maxeidx), min(max(1,eID.GetYaxis().FindBin(pt)),maxeidy))
  sf *= eRECO.GetBinContent(min(max(1,eRECO.GetXaxis().FindBin(eta)),maxerex), min(max(1,eRECO.GetYaxis().FindBin(pt)),maxerey))
  return sf

def getSF2Mu(lep1pt, lep1eta, lep2pt, lep2eta):
  return getSFMu(lep1pt, lep1eta)*getSFMu(lep2pt, lep2eta)

def getSF2El(lep1pt, lep1eta, lep2pt, lep2eta):
  return getSFEl(lep1pt, lep1eta)*getSFEl(lep2pt, lep2eta)


def getSF(nmu, lep1pt, lep1eta, lep2pt, lep2eta):
  if nmu == 2:
    sf = getSF2Mu(lep1pt, lep1eta, lep2pt, lep2eta)
  else:
    sf = getSF2El(lep1pt, lep1eta, lep2pt, lep2eta)
  return sf

def SF(x):
  w = np.array([])
  for i in range(len(x)):
    w = np.append(w, getSF(x["nmuons"][i], x["leadlep_pt"][i], x["leadlep_eta"][i], x["subleadlep_pt"][i], x["subleadlep_eta"][i]))
    #print(w)
  return w


Zptfile = ROOT.TFile(SUEP_BASE + "/data/ZptUL18/Zpt_corr.root","READ")
ZptHist = Zptfile.Get("Zpt_SF")
maxZx   = ZptHist.GetNbinsX()
maxZy   = ZptHist.GetNbinsY()

def correctZpt(x):
  sf = ZptHist.GetBinContent(min(max(1, ZptHist.GetXaxis().FindBin(x["Z_pt"])), maxZx), min(max(1, ZptHist.GetYaxis().FindBin(x["Z_eta"])), maxZy))
  return sf

