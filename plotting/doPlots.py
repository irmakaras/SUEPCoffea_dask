import pandas as pd
import ROOT
import os

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
output = "/eos/user/j/jkil/SUEP/output_graphs"
DY = [pd.HDFStore("../outputDY/"+f, 'r') for f in os.listdir("../outputDY/")]
ZH = [pd.HDFStore("../outputZH/"+f, 'r') for f in os.listdir("../outputZH/")]

channel = "vars"

plots = {
  "Leading Lepton p_{T}": ["leadlep_pt", 50, 0, 200, "p_{T}^{l1} [GeV]"], 
  "Leading Lepton eta":["leadlep_eta", 50, -2.5, 2.5, "eta"],
  "Leading Lepton phi":["leadlep_phi",50,0,6.28,"phi"]
}

for p in plots:
  h1 = ROOT.TH1F(plots[p][0], plots[p][0], plots[p][1], plots[p][2], plots[p][3])
  h2 = h1.Clone(h1.GetName()+ "_2")

  for d in DY:
    for val in d[channel][plots[p][0]]:
      h1.Fill(val)

  for d in ZH:
    for val in d[channel][plots[p][0]]:
      h2.Fill(val)

  theColors = {"1":ROOT.kBlue, "2":ROOT.kRed}
  c = ROOT.TCanvas("c","c", 800,600)
  p1 = ROOT.TPad("mainpad", "mainpad", 0, 0.30, 1, 1)
  p1.SetBottomMargin(0.025)
  p1.SetTopMargin(0.08)
  p1.SetLeftMargin(0.12)
  p1.Draw()
  p1.SetLogy(True)
  p2 = ROOT.TPad("ratiopad", "ratiopad", 0, 0, 1, 0.30)
  p2.SetTopMargin(0.01)
  p2.SetBottomMargin(0.45)
  p2.SetLeftMargin(0.12)
  p2.SetFillStyle(0)
  p2.Draw()

  p1.cd()

  h1.SetTitle("")
  h1.GetYaxis().SetTitle("Normalized events")
  h1.GetYaxis().SetTitleSize(0.05)
  h1.Scale(1./h1.Integral())
  h2.Scale(1./h2.Integral())
  h1.SetLineColor(theColors["1"])
  h2.SetLineColor(theColors["2"])
  h1.SetMaximum(1.1)
  h1.SetMinimum(0.001)
  h1.Draw()
  h2.Draw("same")
  tl = ROOT.TLegend(0.6,0.7,0.9,0.9)
  tl.AddEntry(h1, "DY", "l")
  tl.AddEntry(h2, "ZS, m_{S} = 125 GeV", "l")
  tl.Draw("same")

  p2.cd()
  ratioOff = h1.Clone(h1.GetName().replace("h","r"))
  ratioCus = h2.Clone(h2.GetName().replace("h","r"))
  ratioOff.Divide(h1)
  ratioOff.GetYaxis().SetTitle("Events/DY Events")
  ratioOff.SetTitleSize(0.05)
  ratioCus.Divide(h1)
  ratioCus.SetLineColor(theColors["2"])
  ratioOff.SetMaximum(2.)
  ratioOff.SetMinimum(0.)
  ratioOff.GetXaxis().SetTitle(plots[p][4])
  ratioOff.GetXaxis().SetTitleSize(0.18)
  ratioOff.GetXaxis().SetTitleOffset(1.)
  ratioOff.GetXaxis().SetLabelSize(0.15)
  ratioOff.Draw()
  ratioCus.Draw("same")
  c.SaveAs("%s/%s.pdf"%(output,p))
  c.SaveAs("%s/%s.png"%(output,p))

