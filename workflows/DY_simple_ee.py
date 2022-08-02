"""
SUEP_coffea_ZH.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer, 2021
"""

import os
import pathlib
import shutil
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
from coffea import hist, processor
import vector
from typing import List, Optional
vector.register_awkward()

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, SRonly: bool, output_location: Optional[str], doOF: Optional[bool], isDY: Optional[bool]) -> None:
        self.SRonly = SRonly
        self.output_location = output_location
        self.doOF = doOF
        self.isDY = isDY # We need to save this to remove the overlap between the inclusive DY sample and the pT binned ones
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def sphericity(self, events, particles, r):
        # In principle here we already have ak.num(particles) != 0
        # Some sanity replacements just in case the boosting broke
        px = ak.nan_to_num(particles.px, 0)
        py = ak.nan_to_num(particles.py, 0)
        pz = ak.nan_to_num(particles.pz, 0)
        p  = ak.nan_to_num(particles.p,  0)

        norm = np.squeeze(ak.sum(p ** r, axis=1, keepdims=True))
        s = np.array([[
                       ak.sum(px*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(px*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(py*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(py*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                      ],
                      [
                       ak.sum(pz*px * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*py * p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                       ak.sum(pz*pz * p ** (r-2.0), axis=1 ,keepdims=True)/norm
                       ]])
        s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
        s = np.nan_to_num(s, copy=False, nan=1., posinf=1., neginf=1.) 

        evals = np.sort(np.linalg.eigvals(s))
        # eval1 < eval2 < eval3
        return evals

    def rho(self, number, jet, tracks, deltaR, dr=0.05):
        r_start = number*dr
        r_end = (number+1)*dr
        ring = (deltaR > r_start) & (deltaR < r_end)
        rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
        return rho_values

    def ak_to_pandas(self, jet_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(jet_collection):
            prefix = self.prefixes.get(field, "")
            if len(prefix) > 0:
                for subfield in ak.fields(jet_collection[field]):
                    output[f"{prefix}_{subfield}"] = ak.to_numpy(
                        jet_collection[field][subfield]
                    )
            else:
                if not(isinstance(ak.to_numpy(jet_collection[field])[0],np.ndarray)):
                    output[field] = ak.to_numpy(jet_collection[field])
                else:
                    temp =  ak.to_numpy(jet_collection[field])
                    output[field] = [[k for k in kk] for kk in temp]
        return output

    def h5store(self, store: pd.HDFStore, df: pd.DataFrame, fname: str, gname: str, **kwargs: float) -> None:
        store.put(gname, df)
        store.get_storer(gname).attrs.metadata = kwargs
        
    def save_dfs(self, dfs, df_names, fname=None):
        if not(fname): fname = "out.hdf5"
        subdirs = []
        store = pd.HDFStore(fname)
        if self.output_location is not None:
            # pandas to hdf5
            for out, gname in zip(dfs, df_names):
                if self.isMC:
                    metadata = dict(gensumweight=self.gensumweight,era=self.era, mc=self.isMC,sample=self.sample)
                    #metadata.update({"gensumweight":self.gensumweight})
                else:
                    metadata = dict(era=self.era, mc=self.isMC,sample=self.sample)    
                    
                store_fin = self.h5store(store, out, fname, gname, **metadata)

            store.close()
            self.dump_table(fname, self.output_location, subdirs)
        else:
            print("self.output_location is None")
            store.close()

    def dump_table(self, fname: str, location: str, subdirs: Optional[List[str]] = None) -> None:
        subdirs = subdirs or []
        xrd_prefix = "root://"
        pfx_len = len(xrd_prefix)
        xrootd = False
        if xrd_prefix in location:
            try:
                import XRootD
                import XRootD.client

                xrootd = True
            except ImportError:
                raise ImportError(
                    "Install XRootD python bindings with: conda install -c conda-forge xroot"
                )
        local_file = (
            os.path.abspath(os.path.join(".", fname))
            if xrootd
            else os.path.join(".", fname)
        )
        merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
        destination = (
            location + merged_subdirs + f"/{fname}"
            if xrootd
            else os.path.join(location, os.path.join(merged_subdirs, fname))
        )
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination)
            copyproc.prepare()
            copyproc.run()
            client = XRootD.client.FileSystem(
                location[: location[pfx_len:].find("/") + pfx_len]
            )
            status = client.locate(
                destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
                XRootD.client.flags.OpenFlags.READ,
            )
            assert status[0].ok
            del client
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            if not os.path.exists(dirname):
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            if os.path.isfile(destination):
                if not os.path.samefile(local_file, destination):
                    shutil.copy2(local_file, destination)
                else:
                  fname = "condor_" + fname
                  destination = os.path.join(location, os.path.join(merged_subdirs, fname))
                  shutil.copy2(local_file, destination)
            else:
                shutil.copy2(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


    def selectByTrigger(self, events, extraColls = []):
        ### Apply trigger selection
        ### TODO:: Save a per-event flag that classifies the event (ee or mumu)
        cutAnyHLT = (events.HLT.Ele27_WPTight_Gsf) | (events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)

        return events[cutAnyHLT], [coll[cutAnyHLT] for coll in extraColls]

    def selectByLeptons(self, events, extraColls = []):
    ###lepton selection criteria--4momenta collection for plotting

        muons = ak.zip({
            "pt": events.Muon.pt,
            "eta": events.Muon.eta,
            "phi": events.Muon.phi,
            "mass": events.Muon.mass,
            "charge": events.Muon.pdgId/(-13),
        }, with_name="Momentum4D")
	
        electrons = ak.zip({
            "pt": events.Electron.pt,
            "eta": events.Electron.eta,
            "phi": events.Electron.phi,
            "mass": events.Electron.mass,
            "charge": events.Electron.pdgId/(-11),
        }, with_name="Momentum4D")

        ###  Some very simple selections on ID ###
        ###  Muons: loose ID + dxy dz cuts mimicking the medium prompt ID https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
        ###  Electrons: loose ID + dxy dz cuts for promptness https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCutBasedIdentification
        cutMuons     = (events.Muon.isPFcand) | (events.Muon.isGlobal)
        cutElectrons = (events.Electron.pt >= 10)

        ### Apply the cuts
        # Object selection. selMuons contain only the events that are filtered by cutMuons criteria.
        selMuons     = muons[cutMuons]
        selElectrons = electrons[cutElectrons]

        ### Now global cuts to select events. Notice this means exactly two leptons with pT >= 10, and the leading one pT >= 25

        # cutHasTwoMuons imposes three conditions:
        #  First, number of muons (axis=1 means column. Each row is an event.) in an event is 2.
        #  Second, pt of the muons is greater than 25.
        #  Third, Sum of charge of muons should be 0. (because it originates from Z)
        if True:
            cutHasTwoMuons = (ak.num(selElectrons, axis=1)==2) & (ak.max(selElectrons.pt, axis=1, mask_identity=False) >= 25) & (ak.sum(selElectrons.charge,axis=1) == 0)
            cutTwoLeps     = cutHasTwoMuons
            cutHasTwoLeps  = cutHasTwoMuons
            ### Cut the events, also return the selected leptons for operation down the line
            events = events[cutHasTwoLeps]
            selElectrons = selElectrons[cutHasTwoLeps]
            selMuons = selMuons[cutHasTwoLeps]
          
        return events, selElectrons, selMuons #, [coll[cutHasTwoLeps] for coll in extraColls]

    def selectByJets(self, events, leptons = [],  extraColls = []):
        # These are just standard jets, as available in the nanoAOD
        Jets = ak.zip({
            "pt": events.Jet.pt,
            "eta": events.Jet.eta,
            "phi": events.Jet.phi,
            "mass": events.Jet.mass,
            "btag": events.Jet.btagDeepFlavB,
            "jetId": events.Jet.jetId
        }, with_name="Momentum4D")
        # Minimimum pT, eta requirements + jet-lepton recleaning
        jetCut = (Jets.pt > 30) & (abs(Jets.eta)<4.7) & (Jets.deltaR(leptons[:,0])>= 0.4) & (Jets.deltaR(leptons[:,1])>= 0.4)
        jets = Jets[jetCut]
        # The following is the collection of events and of jets
        return events, jets, [coll for coll in extraColls]
    def selectByMET(self, events):
        MET = ak.zip({
            "pt": events.MET.pt,
            "eta": 0,
            "phi": events.MET.phi,
            "mass": 0,
        })
        puppiMET = ak.zip({
            "pt": events.PuppiMET.pt,
            "eta": 0,
            "phi": events.PuppiMET.phi,
            "mass": 0,
        })
        return events, MET, puppiMET

    def shouldContinueAfterCut(self, events, out):
        #if debug: print("Conversion to pandas...")
        if True: # No need to filter it out
            if len(events) == 0:
                outdfs  = []
                outcols = []
                for channel in out.keys():
                    outcols.append(channel)
                    if out[channel][0] == {}:   
                        outdfs = pd.DataFrame(['empty'], columns=['empty'])
                    else:              
                        if self.isMC:
                            out[channel][0]["genweight"] = out[channel][1].genWeight[:]

                    if not isinstance(out[channel][0], pd.DataFrame): 
                        out[channel][0] = self.ak_to_pandas(out[channel][0])
    
                return False
            else: return True


    def process(self, events):
        #print(events.event[0], events.luminosityBlock[0], events.run[0])
        # 255955082 94729 1
        #if not(events.event[0]==255955082 and events.luminosityBlock[0]==94729 and events.run[0]==1): return self.accumulator.identity()
        if not(self.isMC):
          print("Pre golden", len(events))

          cutGolden = ((events.run == 355381) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 1)) | ((events.luminosityBlock >= 3) & (events.luminosityBlock <= 18)) | ((events.luminosityBlock >= 20) & (events.luminosityBlock <= 20)) | ((events.luminosityBlock >= 22) & (events.luminosityBlock <= 22)) | ((events.luminosityBlock >= 24) & (events.luminosityBlock <= 24)) | ((events.luminosityBlock >= 27) & (events.luminosityBlock <= 27)) | ((events.luminosityBlock >= 36) & (events.luminosityBlock <= 36)) | ((events.luminosityBlock >= 38) & (events.luminosityBlock <= 43)) | ((events.luminosityBlock >= 46) & (events.luminosityBlock <= 48)) | ((events.luminosityBlock >= 53) & (events.luminosityBlock <= 54)) | ((events.luminosityBlock >= 60) & (events.luminosityBlock <= 60)) | ((events.luminosityBlock >= 66) & (events.luminosityBlock <= 66)) | ((events.luminosityBlock >= 68) & (events.luminosityBlock <= 68)) | ((events.luminosityBlock >= 73) & (events.luminosityBlock <= 73)) | ((events.luminosityBlock >= 75) & (events.luminosityBlock <= 78)) | ((events.luminosityBlock >= 80) & (events.luminosityBlock <= 80)) | ((events.luminosityBlock >= 83) & (events.luminosityBlock <= 83)) | ((events.luminosityBlock >= 85) & (events.luminosityBlock <= 85)) | ((events.luminosityBlock >= 87) & (events.luminosityBlock <= 87)) | ((events.luminosityBlock >= 90) & (events.luminosityBlock <= 90)) | ((events.luminosityBlock >= 92) & (events.luminosityBlock <= 92)) | ((events.luminosityBlock >= 94) & (events.luminosityBlock <= 95)) | ((events.luminosityBlock >= 100) & (events.luminosityBlock <= 100)) | ((events.luminosityBlock >= 103) & (events.luminosityBlock <= 122)) | ((events.luminosityBlock >= 124) & (events.luminosityBlock <= 125)) | ((events.luminosityBlock >= 127) & (events.luminosityBlock <= 131)) | ((events.luminosityBlock >= 133) & (events.luminosityBlock <= 135)) | ((events.luminosityBlock >= 137) & (events.luminosityBlock <= 141)) | ((events.luminosityBlock >= 143) & (events.luminosityBlock <= 145)) | ((events.luminosityBlock >= 147) & (events.luminosityBlock <= 149)) | ((events.luminosityBlock >= 151) & (events.luminosityBlock <= 156)) | ((events.luminosityBlock >= 160) & (events.luminosityBlock <= 160)) | ((events.luminosityBlock >= 162) & (events.luminosityBlock <= 162)) | ((events.luminosityBlock >= 165) & (events.luminosityBlock <= 203)) | ((events.luminosityBlock >= 206) & (events.luminosityBlock <= 208)) | ((events.luminosityBlock >= 211) & (events.luminosityBlock <= 218)) | ((events.luminosityBlock >= 222) & (events.luminosityBlock <= 224)) | ((events.luminosityBlock >= 226) & (events.luminosityBlock <= 238)) | ((events.luminosityBlock >= 240) & (events.luminosityBlock <= 240)) | ((events.luminosityBlock >= 243) & (events.luminosityBlock <= 244)) | ((events.luminosityBlock >= 246) & (events.luminosityBlock <= 248)) | ((events.luminosityBlock >= 250) & (events.luminosityBlock <= 292)) | ((events.luminosityBlock >= 296) & (events.luminosityBlock <= 296)) | ((events.luminosityBlock >= 298) & (events.luminosityBlock <= 298)) | ((events.luminosityBlock >= 300) & (events.luminosityBlock <= 317)) | ((events.luminosityBlock >= 319) & (events.luminosityBlock <= 321)) | ((events.luminosityBlock >= 323) & (events.luminosityBlock <= 358))) ) | ((events.run == 355418) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 12)) | ((events.luminosityBlock >= 14) & (events.luminosityBlock <= 14)) | ((events.luminosityBlock >= 17) & (events.luminosityBlock <= 32)) | ((events.luminosityBlock >= 36) & (events.luminosityBlock <= 41))) ) | ((events.run == 355419) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 34)) | ((events.luminosityBlock >= 36) & (events.luminosityBlock <= 98))) ) | ((events.run == 355429) & (((events.luminosityBlock >= 42) & (events.luminosityBlock <= 76)) | ((events.luminosityBlock >= 78) & (events.luminosityBlock <= 78)) | ((events.luminosityBlock >= 82) & (events.luminosityBlock <= 89))) ) | ((events.run == 355435) & (((events.luminosityBlock >= 42) & (events.luminosityBlock <= 84))) ) | ((events.run == 355441) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 3)) | ((events.luminosityBlock >= 5) & (events.luminosityBlock <= 6)) | ((events.luminosityBlock >= 8) & (events.luminosityBlock <= 14))) ) | ((events.run == 355442) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 22))) ) | ((events.run == 355443) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 39)) | ((events.luminosityBlock >= 41) & (events.luminosityBlock <= 120)) | ((events.luminosityBlock >= 122) & (events.luminosityBlock <= 122)) | ((events.luminosityBlock >= 124) & (events.luminosityBlock <= 125)) | ((events.luminosityBlock >= 127) & (events.luminosityBlock <= 131)) | ((events.luminosityBlock >= 133) & (events.luminosityBlock <= 142)) | ((events.luminosityBlock >= 155) & (events.luminosityBlock <= 155)) | ((events.luminosityBlock >= 158) & (events.luminosityBlock <= 158)) | ((events.luminosityBlock >= 160) & (events.luminosityBlock <= 168)) | ((events.luminosityBlock >= 171) & (events.luminosityBlock <= 175)) | ((events.luminosityBlock >= 177) & (events.luminosityBlock <= 178)) | ((events.luminosityBlock >= 180) & (events.luminosityBlock <= 187)) | ((events.luminosityBlock >= 190) & (events.luminosityBlock <= 198)) | ((events.luminosityBlock >= 200) & (events.luminosityBlock <= 206)) | ((events.luminosityBlock >= 208) & (events.luminosityBlock <= 208)) | ((events.luminosityBlock >= 210) & (events.luminosityBlock <= 214)) | ((events.luminosityBlock >= 216) & (events.luminosityBlock <= 220)) | ((events.luminosityBlock >= 223) & (events.luminosityBlock <= 231)) | ((events.luminosityBlock >= 233) & (events.luminosityBlock <= 233))) ) | ((events.run == 355444) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 1)) | ((events.luminosityBlock >= 3) & (events.luminosityBlock <= 3)) | ((events.luminosityBlock >= 5) & (events.luminosityBlock <= 10)) | ((events.luminosityBlock >= 12) & (events.luminosityBlock <= 13)) | ((events.luminosityBlock >= 18) & (events.luminosityBlock <= 19)) | ((events.luminosityBlock >= 23) & (events.luminosityBlock <= 37)) | ((events.luminosityBlock >= 40) & (events.luminosityBlock <= 73)) | ((events.luminosityBlock >= 77) & (events.luminosityBlock <= 78)) | ((events.luminosityBlock >= 80) & (events.luminosityBlock <= 143)) | ((events.luminosityBlock >= 145) & (events.luminosityBlock <= 146)) | ((events.luminosityBlock >= 148) & (events.luminosityBlock <= 149)) | ((events.luminosityBlock >= 151) & (events.luminosityBlock <= 153))) ) | ((events.run == 355445) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 51)) | ((events.luminosityBlock >= 53) & (events.luminosityBlock <= 62)) | ((events.luminosityBlock >= 65) & (events.luminosityBlock <= 151)) | ((events.luminosityBlock >= 153) & (events.luminosityBlock <= 242))) ) | ((events.run == 355454) & (((events.luminosityBlock >= 38) & (events.luminosityBlock <= 118))) ) | ((events.run == 355455) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 40))) ) | ((events.run == 355456) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 24)) | ((events.luminosityBlock >= 27) & (events.luminosityBlock <= 92)) | ((events.luminosityBlock >= 95) & (events.luminosityBlock <= 433)) | ((events.luminosityBlock >= 436) & (events.luminosityBlock <= 438)) | ((events.luminosityBlock >= 440) & (events.luminosityBlock <= 451)) | ((events.luminosityBlock >= 453) & (events.luminosityBlock <= 454)) | ((events.luminosityBlock >= 456) & (events.luminosityBlock <= 501))) ) | ((events.run == 355559) & (((events.luminosityBlock >= 35) & (events.luminosityBlock <= 35)) | ((events.luminosityBlock >= 38) & (events.luminosityBlock <= 39))) ) | ((events.run == 355679) & (((events.luminosityBlock >= 66) & (events.luminosityBlock <= 66)) | ((events.luminosityBlock >= 68) & (events.luminosityBlock <= 85))) ) | ((events.run == 355680) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 87)) | ((events.luminosityBlock >= 89) & (events.luminosityBlock <= 135)) | ((events.luminosityBlock >= 137) & (events.luminosityBlock <= 137)) | ((events.luminosityBlock >= 139) & (events.luminosityBlock <= 139)) | ((events.luminosityBlock >= 141) & (events.luminosityBlock <= 145)) | ((events.luminosityBlock >= 147) & (events.luminosityBlock <= 153)) | ((events.luminosityBlock >= 156) & (events.luminosityBlock <= 157)) | ((events.luminosityBlock >= 161) & (events.luminosityBlock <= 188)) | ((events.luminosityBlock >= 190) & (events.luminosityBlock <= 190)) | ((events.luminosityBlock >= 193) & (events.luminosityBlock <= 194)) | ((events.luminosityBlock >= 196) & (events.luminosityBlock <= 252)) | ((events.luminosityBlock >= 254) & (events.luminosityBlock <= 256)) | ((events.luminosityBlock >= 259) & (events.luminosityBlock <= 266)) | ((events.luminosityBlock >= 268) & (events.luminosityBlock <= 291)) | ((events.luminosityBlock >= 293) & (events.luminosityBlock <= 300)) | ((events.luminosityBlock >= 305) & (events.luminosityBlock <= 305)) | ((events.luminosityBlock >= 308) & (events.luminosityBlock <= 316)) | ((events.luminosityBlock >= 318) & (events.luminosityBlock <= 318)) | ((events.luminosityBlock >= 320) & (events.luminosityBlock <= 320)) | ((events.luminosityBlock >= 322) & (events.luminosityBlock <= 323)) | ((events.luminosityBlock >= 325) & (events.luminosityBlock <= 331)) | ((events.luminosityBlock >= 333) & (events.luminosityBlock <= 348)) | ((events.luminosityBlock >= 358) & (events.luminosityBlock <= 398)) | ((events.luminosityBlock >= 400) & (events.luminosityBlock <= 401)) | ((events.luminosityBlock >= 404) & (events.luminosityBlock <= 407)) | ((events.luminosityBlock >= 409) & (events.luminosityBlock <= 413)) | ((events.luminosityBlock >= 416) & (events.luminosityBlock <= 418)) | ((events.luminosityBlock >= 420) & (events.luminosityBlock <= 425)) | ((events.luminosityBlock >= 427) & (events.luminosityBlock <= 433)) | ((events.luminosityBlock >= 435) & (events.luminosityBlock <= 463)) | ((events.luminosityBlock >= 465) & (events.luminosityBlock <= 465)) | ((events.luminosityBlock >= 467) & (events.luminosityBlock <= 467)) | ((events.luminosityBlock >= 469) & (events.luminosityBlock <= 508)) | ((events.luminosityBlock >= 510) & (events.luminosityBlock <= 512)) | ((events.luminosityBlock >= 514) & (events.luminosityBlock <= 515)) | ((events.luminosityBlock >= 519) & (events.luminosityBlock <= 520)) | ((events.luminosityBlock >= 523) & (events.luminosityBlock <= 525)) | ((events.luminosityBlock >= 552) & (events.luminosityBlock <= 552)) | ((events.luminosityBlock >= 555) & (events.luminosityBlock <= 557)) | ((events.luminosityBlock >= 564) & (events.luminosityBlock <= 564)) | ((events.luminosityBlock >= 566) & (events.luminosityBlock <= 566)) | ((events.luminosityBlock >= 568) & (events.luminosityBlock <= 571)) | ((events.luminosityBlock >= 573) & (events.luminosityBlock <= 573)) | ((events.luminosityBlock >= 588) & (events.luminosityBlock <= 589)) | ((events.luminosityBlock >= 591) & (events.luminosityBlock <= 593)) | ((events.luminosityBlock >= 595) & (events.luminosityBlock <= 598)) | ((events.luminosityBlock >= 601) & (events.luminosityBlock <= 603)) | ((events.luminosityBlock >= 606) & (events.luminosityBlock <= 607)) | ((events.luminosityBlock >= 619) & (events.luminosityBlock <= 619)) | ((events.luminosityBlock >= 623) & (events.luminosityBlock <= 623)) | ((events.luminosityBlock >= 628) & (events.luminosityBlock <= 629)) | ((events.luminosityBlock >= 639) & (events.luminosityBlock <= 639)) | ((events.luminosityBlock >= 641) & (events.luminosityBlock <= 641)) | ((events.luminosityBlock >= 643) & (events.luminosityBlock <= 645)) | ((events.luminosityBlock >= 651) & (events.luminosityBlock <= 651)) | ((events.luminosityBlock >= 654) & (events.luminosityBlock <= 654)) | ((events.luminosityBlock >= 659) & (events.luminosityBlock <= 659)) | ((events.luminosityBlock >= 672) & (events.luminosityBlock <= 672)) | ((events.luminosityBlock >= 675) & (events.luminosityBlock <= 675)) | ((events.luminosityBlock >= 679) & (events.luminosityBlock <= 679)) | ((events.luminosityBlock >= 683) & (events.luminosityBlock <= 685)) | ((events.luminosityBlock >= 690) & (events.luminosityBlock <= 690)) | ((events.luminosityBlock >= 693) & (events.luminosityBlock <= 693)) | ((events.luminosityBlock >= 696) & (events.luminosityBlock <= 696)) | ((events.luminosityBlock >= 701) & (events.luminosityBlock <= 701)) | ((events.luminosityBlock >= 707) & (events.luminosityBlock <= 708)) | ((events.luminosityBlock >= 711) & (events.luminosityBlock <= 711)) | ((events.luminosityBlock >= 713) & (events.luminosityBlock <= 713)) | ((events.luminosityBlock >= 715) & (events.luminosityBlock <= 716)) | ((events.luminosityBlock >= 719) & (events.luminosityBlock <= 731)) | ((events.luminosityBlock >= 733) & (events.luminosityBlock <= 733)) | ((events.luminosityBlock >= 736) & (events.luminosityBlock <= 739)) | ((events.luminosityBlock >= 742) & (events.luminosityBlock <= 742)) | ((events.luminosityBlock >= 747) & (events.luminosityBlock <= 747)) | ((events.luminosityBlock >= 750) & (events.luminosityBlock <= 752)) | ((events.luminosityBlock >= 754) & (events.luminosityBlock <= 754)) | ((events.luminosityBlock >= 760) & (events.luminosityBlock <= 760)) | ((events.luminosityBlock >= 762) & (events.luminosityBlock <= 762)) | ((events.luminosityBlock >= 764) & (events.luminosityBlock <= 764)) | ((events.luminosityBlock >= 767) & (events.luminosityBlock <= 768)) | ((events.luminosityBlock >= 771) & (events.luminosityBlock <= 771)) | ((events.luminosityBlock >= 775) & (events.luminosityBlock <= 777)) | ((events.luminosityBlock >= 780) & (events.luminosityBlock <= 780)) | ((events.luminosityBlock >= 784) & (events.luminosityBlock <= 784)) | ((events.luminosityBlock >= 788) & (events.luminosityBlock <= 788)) | ((events.luminosityBlock >= 791) & (events.luminosityBlock <= 791)) | ((events.luminosityBlock >= 794) & (events.luminosityBlock <= 795)) | ((events.luminosityBlock >= 798) & (events.luminosityBlock <= 798)) | ((events.luminosityBlock >= 800) & (events.luminosityBlock <= 800)) | ((events.luminosityBlock >= 802) & (events.luminosityBlock <= 802)) | ((events.luminosityBlock >= 804) & (events.luminosityBlock <= 805)) | ((events.luminosityBlock >= 807) & (events.luminosityBlock <= 808)) | ((events.luminosityBlock >= 812) & (events.luminosityBlock <= 814)) | ((events.luminosityBlock >= 903) & (events.luminosityBlock <= 913)) | ((events.luminosityBlock >= 915) & (events.luminosityBlock <= 949)) | ((events.luminosityBlock >= 951) & (events.luminosityBlock <= 951)) | ((events.luminosityBlock >= 956) & (events.luminosityBlock <= 957)) | ((events.luminosityBlock >= 959) & (events.luminosityBlock <= 959)) | ((events.luminosityBlock >= 961) & (events.luminosityBlock <= 973)) | ((events.luminosityBlock >= 977) & (events.luminosityBlock <= 977)) | ((events.luminosityBlock >= 979) & (events.luminosityBlock <= 980)) | ((events.luminosityBlock >= 982) & (events.luminosityBlock <= 996)) | ((events.luminosityBlock >= 999) & (events.luminosityBlock <= 1002)) | ((events.luminosityBlock >= 1007) & (events.luminosityBlock <= 1009)) | ((events.luminosityBlock >= 1011) & (events.luminosityBlock <= 1012)) | ((events.luminosityBlock >= 1014) & (events.luminosityBlock <= 1015)) | ((events.luminosityBlock >= 1018) & (events.luminosityBlock <= 1018)) | ((events.luminosityBlock >= 1021) & (events.luminosityBlock <= 1052)) | ((events.luminosityBlock >= 1054) & (events.luminosityBlock <= 1086)) | ((events.luminosityBlock >= 1088) & (events.luminosityBlock <= 1120)) | ((events.luminosityBlock >= 1123) & (events.luminosityBlock <= 1123)) | ((events.luminosityBlock >= 1125) & (events.luminosityBlock <= 1146)) | ((events.luminosityBlock >= 1148) & (events.luminosityBlock <= 1150)) | ((events.luminosityBlock >= 1153) & (events.luminosityBlock <= 1193)) | ((events.luminosityBlock >= 1195) & (events.luminosityBlock <= 1195)) | ((events.luminosityBlock >= 1197) & (events.luminosityBlock <= 1229)) | ((events.luminosityBlock >= 1231) & (events.luminosityBlock <= 1241)) | ((events.luminosityBlock >= 1247) & (events.luminosityBlock <= 1249)) | ((events.luminosityBlock >= 1251) & (events.luminosityBlock <= 1251)) | ((events.luminosityBlock >= 1257) & (events.luminosityBlock <= 1297)) | ((events.luminosityBlock >= 1299) & (events.luminosityBlock <= 1360)) | ((events.luminosityBlock >= 1370) & (events.luminosityBlock <= 1370)) | ((events.luminosityBlock >= 1391) & (events.luminosityBlock <= 1395)) | ((events.luminosityBlock >= 1397) & (events.luminosityBlock <= 1398)) | ((events.luminosityBlock >= 1400) & (events.luminosityBlock <= 1400)) | ((events.luminosityBlock >= 1404) & (events.luminosityBlock <= 1404)) | ((events.luminosityBlock >= 1406) & (events.luminosityBlock <= 1407)) | ((events.luminosityBlock >= 1409) & (events.luminosityBlock <= 1415)) | ((events.luminosityBlock >= 1417) & (events.luminosityBlock <= 1418)) | ((events.luminosityBlock >= 1420) & (events.luminosityBlock <= 1421)) | ((events.luminosityBlock >= 1423) & (events.luminosityBlock <= 1424)) | ((events.luminosityBlock >= 1429) & (events.luminosityBlock <= 1431)) | ((events.luminosityBlock >= 1433) & (events.luminosityBlock <= 1433)) | ((events.luminosityBlock >= 1435) & (events.luminosityBlock <= 1435)) | ((events.luminosityBlock >= 1437) & (events.luminosityBlock <= 1437)) | ((events.luminosityBlock >= 1439) & (events.luminosityBlock <= 1441)) | ((events.luminosityBlock >= 1446) & (events.luminosityBlock <= 1447)) | ((events.luminosityBlock >= 1449) & (events.luminosityBlock <= 1450)) | ((events.luminosityBlock >= 1452) & (events.luminosityBlock <= 1455)) | ((events.luminosityBlock >= 1457) & (events.luminosityBlock <= 1458)) | ((events.luminosityBlock >= 1460) & (events.luminosityBlock <= 1460)) | ((events.luminosityBlock >= 1463) & (events.luminosityBlock <= 1465)) | ((events.luminosityBlock >= 1468) & (events.luminosityBlock <= 1468)) | ((events.luminosityBlock >= 1472) & (events.luminosityBlock <= 1474)) | ((events.luminosityBlock >= 1477) & (events.luminosityBlock <= 1479)) | ((events.luminosityBlock >= 1483) & (events.luminosityBlock <= 1483)) | ((events.luminosityBlock >= 1489) & (events.luminosityBlock <= 1492)) | ((events.luminosityBlock >= 1494) & (events.luminosityBlock <= 1494)) | ((events.luminosityBlock >= 1496) & (events.luminosityBlock <= 1498)) | ((events.luminosityBlock >= 1503) & (events.luminosityBlock <= 1503)) | ((events.luminosityBlock >= 1507) & (events.luminosityBlock <= 1512)) | ((events.luminosityBlock >= 1514) & (events.luminosityBlock <= 1515)) | ((events.luminosityBlock >= 1518) & (events.luminosityBlock <= 1519)) | ((events.luminosityBlock >= 1524) & (events.luminosityBlock <= 1526)) | ((events.luminosityBlock >= 1532) & (events.luminosityBlock <= 1533))) ) | ((events.run == 355768) & (((events.luminosityBlock >= 82) & (events.luminosityBlock <= 126))) ) | ((events.run == 355769) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 106)) | ((events.luminosityBlock >= 108) & (events.luminosityBlock <= 188)) | ((events.luminosityBlock >= 191) & (events.luminosityBlock <= 191)) | ((events.luminosityBlock >= 194) & (events.luminosityBlock <= 194)) | ((events.luminosityBlock >= 196) & (events.luminosityBlock <= 196)) | ((events.luminosityBlock >= 198) & (events.luminosityBlock <= 316)) | ((events.luminosityBlock >= 319) & (events.luminosityBlock <= 320)) | ((events.luminosityBlock >= 324) & (events.luminosityBlock <= 502)) | ((events.luminosityBlock >= 504) & (events.luminosityBlock <= 506)) | ((events.luminosityBlock >= 508) & (events.luminosityBlock <= 535)) | ((events.luminosityBlock >= 537) & (events.luminosityBlock <= 537)) | ((events.luminosityBlock >= 540) & (events.luminosityBlock <= 541))) ) | ((events.run == 355862) & (((events.luminosityBlock >= 121) & (events.luminosityBlock <= 126))) ) | ((events.run == 355863) & (((events.luminosityBlock >= 1) & (events.luminosityBlock <= 14))) )
          events = events[cutGolden]
          print("Post golden", len(events)) 
        debug    = True  # If we want some prints in the middle
        chunkTag = "out_%i_%i_%i.hdf5"%(events.event[0], events.luminosityBlock[0], events.run[0]) #Unique tag to get different outputs per tag
        fullFile = self.output_location + "/" + chunkTag
        print("Check file %s"%fullFile)
        if os.path.isfile(fullFile): 
            print("SKIP")
            return self.accumulator.identity()
        # Main processor code


        # ------------------------------------------------------------------------------------
        # ------------------------------- DEFINE OUTPUTS -------------------------------------
        # ------------------------------------------------------------------------------------

        accumulator    = self.accumulator.identity()
        # Each track is one selection level
        outputs = {
            "twoleptons"  :[{},[]],   # Has Two Leptons, pT and Trigger requirements
        }

        # Data dependant stuff
        dataset = events.metadata['dataset']
        if self.isMC: self.gensumweight = ak.sum(events.genWeight)

        if not(self.isMC): doGen = False

        # ------------------------------------------------------------------------------------
        # ------------------------------- OBJECT LOADING -------------------------------------
        # ------------------------------------------------------------------------------------

        # Lepton selection
        if debug: print("Applying lepton requirements.... %i events in"%len(events))
        self.events, self.electrons, self.muons = self.selectByLeptons(events)[:3]
        print(len(self.events), len(self.electrons), len(self.muons))
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator # If we have no events, we simply stop
        # Trigger selection
        if debug: print("%i events pass lepton cuts. Applying trigger requirements...."%len(self.events))
        self.events, [self.electrons, self.muons] = self.selectByTrigger(self.events,[self.electrons, self.muons])
        print(len(self.events), len(self.electrons), len(self.muons))
        # Here we join muons and electrons into leptons and sort them by pT
        self.leptons = ak.concatenate([self.electrons, self.muons], axis=1)
        highpt_leptons = ak.argsort(self.leptons.pt, axis=1, ascending=False, stable=True)
        self.leptons = self.leptons[highpt_leptons]
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator
        if debug: print("%i events pass trigger cuts. Selecting jets..."%len(self.events))

        # Now do jet selection, for the moment no jet cuts
        self.events, self.jets = self.selectByJets(self.events, self.leptons)[:2] # Leptons are needed to do jet-lepton cleaning
	# Sorting jets by pt.
        highpt_jets = ak.argsort(self.jets.pt, axis=1, ascending=False, stable=True)
        self.jets   = self.jets[highpt_jets]
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator
        if debug: print("%i events pass jet cuts. Selecting tracks..."%len(self.events))
        
        # Now do jet selection, for the moment no jet cuts
        self.events, self.met, self.puppimet = self.selectByMET(self.events)[:3] 
        if not(self.shouldContinueAfterCut(self.events, outputs)): return accumulator
        if debug: print("%i events pass MET cuts. Selecting tracks..."%len(self.events))


        # Now deal with the Z candidate
        self.Zcands = self.leptons[:,0] + self.leptons[:,1]
        
        # ------------------------------------------------------------------------------
        # ------------------------------- SELECTION + PLOTTING -------------------------
        # ------------------------------------------------------------------------------
        outputs["twoleptons"] = [self.doAllPlots("twoleptons", debug), self.events]

        # ------------------------------------------------------------------------------
        # -------------------------------- SAVING --------------------------------------
        # ------------------------------------------------------------------------------
        todel = []
        if self.SRonly: # Lightweight, save only SR stuff
            for out in outputs:
                if not("SR"==out): 
                    todel.append(out)
            for t in todel:
                del outputs[t]

        for out in outputs:
            if out in todel: continue 
            if self.isMC:
                outputs[out][0]["genweight"] = outputs[out][1].genWeight[:]
            if debug: print("Conversion to pandas...")
            if not isinstance(outputs[out][0], pd.DataFrame):
                if debug: print("......%s"%out)
                outputs[out][0] = self.ak_to_pandas(outputs[out][0])

        if debug: print("DFS saving....")

        self.save_dfs([outputs[key][0] for key in outputs], [key for key in outputs], chunkTag)

        return accumulator
   

    def applyCutToAllCollections(self, cut): # Cut has to by a selection applicable across all collections, i.e. something defined per event
        self.events    = self.events[cut]
        self.electrons = self.electrons[cut]
        self.muons     = self.muons[cut]
        self.leptons   = self.leptons[cut]
        self.jets      = self.jets[cut]
        self.Zcands    = self.Zcands[cut]
        self.met       = self.met[cut]
        self.puppimet  = self.puppimet[cut]

    def doAllPlots(self, channel, debug=True):
        # ------------------------------------------------------------------------------
        # ------------------------------- PLOTTING -------------------------------------
        # ------------------------------------------------------------------------------
        out = {}
        # Define outputs for plotting
        if debug: print("Saving reco variables for channel %s"%channel)

        # Object: leptons
        out["leadlep_pt"]    = self.leptons.pt[:,0]
        out["subleadlep_pt"] = self.leptons.pt[:,1]
        out["leadlep_eta"]   = self.leptons.eta[:,0]
        out["subleadlep_eta"]= self.leptons.eta[:,1]
        out["leadlep_phi"]   = self.leptons.phi[:,0]
        out["subleadlep_phi"]= self.leptons.phi[:,1]
        out["nleptons"]      = ak.num(self.leptons, axis=1)[:]
        out["nmuons"]        = ak.num(self.muons) 
        out["nelectrons"]    = ak.num(self.electrons)


        # Object: reconstructed Z
        out["Z_pt"]  = self.Zcands.pt[:]
        out["Z_eta"] = self.Zcands.eta[:]
        out["Z_phi"] = self.Zcands.phi[:]
        out["Z_m"]   = self.Zcands.mass[:]
        
        # Object: jets, a bit tricky as number varies per event!
        out["njets"]          = ak.num(self.jets, axis=1)[:]
        out["nBLoose"]        = ak.sum((self.jets.btag >= 0.0490), axis=1)[:]
        out["nBMedium"]       = ak.sum((self.jets.btag >= 0.2783), axis=1)[:]
        out["nBTight"]        = ak.sum((self.jets.btag >= 0.7100), axis=1)[:]
        out["leadjet_pt"]     = ak.fill_none(ak.pad_none(self.jets.pt,  1, axis=1, clip=True), 0.)[:,0] # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["leadjet_eta"]    = ak.fill_none(ak.pad_none(self.jets.eta, 1, axis=1, clip=True), -999)[:,0] # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["leadjet_phi"]    = ak.fill_none(ak.pad_none(self.jets.phi, 1, axis=1, clip=True), -999)[:,0] # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["subleadjet_pt"]  = ak.fill_none(ak.pad_none(self.jets.pt,  2, axis=1, clip=True), 0.)[:,1] # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["subleadjet_eta"] = ak.fill_none(ak.pad_none(self.jets.eta, 2, axis=1, clip=True), -999)[:,1] # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["subleadjet_phi"] = ak.fill_none(ak.pad_none(self.jets.phi, 2, axis=1, clip=True), -999)[:,1] # So take all events, if there is no jet_pt fill it with none, then replace none with -999

        out["trailjet_pt"]    = ak.fill_none(ak.pad_none(self.jets.pt,  3, axis=1, clip=True), 0.)[:,2] # So take all events, if there is no jet_pt fill it with none, then replace none with 0
        out["trailjet_eta"]   = ak.fill_none(ak.pad_none(self.jets.eta, 3, axis=1, clip=True), -999)[:,2] # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["trailjet_phi"]   = ak.fill_none(ak.pad_none(self.jets.phi, 3, axis=1, clip=True), -999)[:,2] # So take all events, if there is no jet_pt fill it with none, then replace none with -999
        out["H_T"]            = ak.sum(self.jets.pt, axis=1)[:]
        out["L_T"]            = ak.sum(self.leptons.pt, axis=1)[:]
        out["MET"]            = self.met.pt[:]
        out["puppiMET"]       = self.puppimet.pt[:]
        out["MET_phi"]        = self.met.phi[:]
        out["puppiMET_phi"]   = self.puppimet.phi[:]
        if self.isMC: out["Pileup_nTrueInt"] = self.events.Pileup.nTrueInt[:]
        return out

    def postprocess(self, accumulator):
        return accumulator
 
