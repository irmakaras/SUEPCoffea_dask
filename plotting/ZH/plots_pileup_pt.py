import ROOT

def cut(x):
  return (x["njets"] >= 0) 

plots = {
  "ntracks": {
             "name"     : "ntracks",
             "bins"     : ["uniform", 10, 0, 200],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["ntracks"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .00001,
             "ratiomaxY": 1.5,
             "ratiominY": 1e-9,
             "plotname" : "ntracks",
             "xlabel"   : "N_{tracks}",
             "vars"     : ["ntracks"]
  },
    

    
      "njets": {
             "name"     : "njets",
             "bins"     : ["uniform", 15, 0, 20],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["njets"], y*cut(x)),
             "logY"     : True,
             "normalize": True, 
             "maxY"     : 100,
             "minY"     : 0.00001,
             "ratiomaxY": 2.,
             "ratiominY": 0.001,
             "plotname" : "njets",
             "xlabel"   : "N_{jets}",
             "vars"     : ["njets"]
  },
  
      "HT": {
             "name"     : "HT",
             "bins"     : ["uniform", 15, 0, 500],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["H_T"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .00001,
             "ratiomaxY": 2.,
             "ratiominY": 0.001,
             "plotname" : "HT",
             "xlabel"   : "H_{T}(p_{T}^{jet} > 30 GeV)",
             "vars"     : ["leadjet_pt", "subleadjet_pt", "trailjet_pt"]
  },
  "leadclustertracks": {
             "name"     : "leadclustertracks",
             "bins"     : ["uniform", 10, 0, 200],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_ntracks"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .000001,
             "ratiomaxY": 2.,
             "ratiominY": 0,
             "plotname" : "leadclustertracks",
             "xlabel"   : "N_{tracks}^{SUEP}",
             "vars"     : ["leadcluster_ntracks"]
  },
    
      "jet1pt": {
             "name"     : "jet1pt",
             "bins"     : ["uniform", 20, 0, 200],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadjet_pt"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2.,
             "ratiominY": 0.001,
             "plotname" : "jet1pt",
             "xlabel"   : "p_{T}^{jet1}",
             "vars"     : ["leadjet_pt"]
  },
    
    
  "leadclusterpt": {
             "name"     : "leadclusterpt",
             "bins"     : ["uniform", 15, 0, 200],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadcluster_pt"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclusterpt",
             "xlabel"   : "p_{T}^{SUEP} [GeV]",
             "vars"     : ["leadcluster_pt"]
  },
      "Zpt": {
             "name"     : "Zpt",
             "bins"     : ["uniform", 10, 0, 200],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["Z_pt"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "Zpt",
             "xlabel"   : "p_{T}^{Z} [GeV]",
             "vars"     : ["Z_pt"]
  },
      "Zphi": {
             "name"     : "Zphi",
             "bins"     : ["uniform", 10, -3.14, 3.14],
             "channel"  : "twoleptons",
             "value"    : lambda x, y : (x["Z_phi"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : 0.001,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "Zphi",
             "xlabel"   : "#phi^{Z}",
             "vars"     : ["Z_phi"]
  },
      "Zeta": {
             "name"     : "Zeta",
             "bins"     : ["uniform", 10, -5, 5],
             "channel"  : "twoleptons",
             "value"    : lambda x, y : (x["Z_eta"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "Zeta",
             "xlabel"   : "#eta^{Z}",
             "vars"     : ["Z_eta"]
  },
   
      "leadclusterspher": {
             "name"     : "leadclusterspher",
             "bins"     : ["uniform", 10, 0, 1],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadclusterSpher_C"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2,
             "ratiominY": 0.,
             "plotname" : "leadclusterspher",
             "xlabel"   : "S^{SUEP}",
             "vars"     : ["leadclusterSpher_C"]
  },
      "leadclusterspherlab": {
             "name"     : "leadclusterspherlab",
             "bins"     : ["uniform", 10, 0, 1],
             "channel"  : "onecluster",
             "value"    : lambda x, y : (x["leadclusterSpher_L"], y*cut(x)),
             "logY"     : True,
             "normalize": True,
             "maxY"     : 100,
             "minY"     : .001,
             "ratiomaxY": 2.,
             "ratiominY": 0.,
             "plotname" : "leadclusterspherlab",
             "xlabel"   : "S^{SUEP}_{lab}",
             "vars"     : ["leadclusterSpher_L"]
  },
}

 