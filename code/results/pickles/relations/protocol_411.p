�]q (c__main__
Relation
q)�q}q(X   sent_idxqK X   arg1qK K�qX   arg2qKK�qX   pq	c__main__
ProtoFile
q
)�q}q(X   filenameqXL   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/dev_full/protocol_411qX   basenameqX   protocol_411qX   protocol_nameqhX	   text_fileqXP   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/dev_full/protocol_411.txtqX   ann_fileqXP   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/dev_full/protocol_411.annqX	   tokenizerqcsacremoses.tokenize
MosesTokenizer
q)�q}q(X   langqX   enqX   NONBREAKING_PREFIXESq]q(X   AqX   BqX   Cq X   Dq!X   Eq"X   Fq#X   Gq$X   Hq%X   Iq&X   Jq'X   Kq(X   Lq)X   Mq*X   Nq+X   Oq,X   Pq-X   Qq.X   Rq/X   Sq0X   Tq1X   Uq2X   Vq3X   Wq4X   Xq5X   Yq6X   Zq7X   Adjq8X   Admq9X   Advq:X   Asstq;X   Bartq<X   Bldgq=X   Brigq>X   Brosq?X   Captq@X   CmdrqAX   ColqBX   ComdrqCX   ConqDX   CorpqEX   CplqFX   DRqGX   DrqHX   DrsqIX   EnsqJX   GenqKX   GovqLX   HonqMX   HrqNX   HospqOX   InspqPX   LtqQX   MMqRX   MRqSX   MRSqTX   MSqUX   MajqVX   MessrsqWX   MlleqXX   MmeqYX   MrqZX   Mrsq[X   Msq\X   Msgrq]X   Opq^X   Ordq_X   Pfcq`X   PhqaX   ProfqbX   PvtqcX   RepqdX   RepsqeX   ResqfX   RevqgX   RtqhX   SenqiX   SensqjX   SfcqkX   SgtqlX   SrqmX   StqnX   SuptqoX   SurgqpX   vqqX   vsqrX   i.eqsX   revqtX   e.gquX   No #NUMERIC_ONLY#qvX   NosqwX   Art #NUMERIC_ONLY#qxX   NrqyX   pp #NUMERIC_ONLY#qzX   Janq{X   Febq|X   Marq}X   Aprq~X   JunqX   Julq�X   Augq�X   Sepq�X   Octq�X   Novq�X   Decq�eX   NUMERIC_ONLY_PREFIXESq�]q�(X   Noq�X   Artq�X   ppq�eubX   linesq�]q�(XX   Investigating non-specific protein binding and streptavidin binding using QCM-D sensing
q�XW   Prepare a Hepes Buffered Saline (HBS) solution containing 150 mM NaCl and 10 mM Hepes.
q�X;   Then prepare a solution of 25 µg/ml streptavidin in HBS.
q�X:   Using QCM-D sensing, make a baseline of the HBS solution.
q�X�   Expose the modified sensors to non-diluted Fetal Bovine Serum (FBS) for 30-60 min under static conditions in QCM-D and measure the amount of adsorbed FBS relative to the HBS baseline obtained in the previous step.
q�X)   Rinse the sensors with the HBS solution.
q�Xg   Flow the streptavidin solution over the modified sensors (100-500 µl/min), until they are saturated.
q�X)   Rinse the sensors with the HBS solution.
q�eX   textq�X�  Investigating non-specific protein binding and streptavidin binding using QCM-D sensing
Prepare a Hepes Buffered Saline (HBS) solution containing 150 mM NaCl and 10 mM Hepes.
Then prepare a solution of 25 µg/ml streptavidin in HBS.
Using QCM-D sensing, make a baseline of the HBS solution.
Expose the modified sensors to non-diluted Fetal Bovine Serum (FBS) for 30-60 min under static conditions in QCM-D and measure the amount of adsorbed FBS relative to the HBS baseline obtained in the previous step.
Rinse the sensors with the HBS solution.
Flow the streptavidin solution over the modified sensors (100-500 µl/min), until they are saturated.
Rinse the sensors with the HBS solution.
q�X   annq�]q�(X   T1	Action 88 95	Prepare
q�X0   E1 Action:T1 Product:T9 Acts-on:T11 Acts-on2:T13q�X   T2	Action 180 187	prepare
q�X   E2 Action:T2 Product:T16q�X"   T3	Action 253 268	make a baseline
q�X   E3 Action:T3 Acts-on:T14q�X   T4	Action 290 296	Expose
q�X6   E4 Action:T4 Acts-on:T20 Site:T22 Using:T24 Using2:T25q�X   T5	Action 409 416	measure
q�X   T6	Action 504 509	Rinse
q�X"   E6 Action:T6 Acts-on:T29 Using:T30q�X   T7	Action 545 549	Flow
q�X!   E7 Action:T7 Acts-on:T31 Site:T33q�X   T8	Action 646 651	Rinse
q�X"   E8 Action:T8 Acts-on:T35 Using:T36q�X(   T9	Reagent 98 119	Hepes Buffered Saline
q�X!   T10	Concentration 146 152	150 mM
q�X   T11	Reagent 153 157	NaCl
q�X    T12	Concentration 162 167	10 mM
q�X   T13	Reagent 168 173	Hepes
q�X   R1 Measure Arg1:T13 Arg2:T12q�X   R2 Measure Arg1:T11 Arg2:T10q�X$   T15	Concentration 202 210	25 µg/ml
q�X!   T16	Reagent 211 223	streptavidin
q�X   T17	Reagent 227 230	HBS
q�X   R3 Measure Arg1:T16 Arg2:T15q�X   R4 Meronym Arg1:T16 Arg2:T17q�X!   T14	Reagent 276 288	HBS solution
q�X   T19	Modifier 301 309	modified
q�X   T20	Device 310 317	sensors
q�X!   T21	Modifier 321 332	non-diluted
q�X'   T22	Reagent 333 351	Fetal Bovine Serum
q�X   T23	Time 362 371	30-60 min
q�X%   T24	Method 378 395	static conditions
q�X   T25	Device 399 404	QCM-D
q�X   T26	Amount 421 427	amount
q�X   R7 Mod-Link Arg1:T22 Arg2:T21q�X   R8 Mod-Link Arg1:T20 Arg2:T19q�X   R9 Setting Arg1:T4 Arg2:T23q�X   T29	Device 514 521	sensors
q�X!   T30	Reagent 531 543	HBS solution
q�X*   T31	Reagent 554 575	streptavidin solution
q�X   T32	Modifier 585 593	modified
q�X   T33	Device 594 601	sensors
q�X,   T34	Generic-Measure 603 617	100-500 µl/min
q�X   R10 Mod-Link Arg1:T33 Arg2:T32q�X   R11 Measure Arg1:T31 Arg2:T34q�X   T35	Device 656 663	sensors
q�X!   T36	Reagent 673 685	HBS solution
q�X   T37	Action 0 13	Investigating
q�X&   E9 Action:T37 Acts-on:T39 Acts-on2:T40q�X    T38	Modifier 14 26	non-specific
q�X"   T39	Reagent 27 42	protein binding
q�X'   T40	Reagent 47 67	streptavidin binding
q�X   T41	Action 74 87	QCM-D sensing
q�X   E10 Action:T41q�X   T42	Reagent 121 124	HBS
q�X   T43	Reagent 126 134	solution
q�X%   R12 Coreference-Link Arg1:T42 Arg2:T9q�X   R13 Mod-Link Arg1:T39 Arg2:T38q�X   R14 Meronym Arg1:T11 Arg2:T9q�X   R15 Meronym Arg1:T13 Arg2:T9q�X   T44	Reagent 190 198	solution
q�X   T45	Reagent 353 356	FBS
q�X&   R16 Coreference-Link Arg1:T45 Arg2:T22q�X   T46	Reagent 440 444	FBS 
q�X!   T18	Action 238 251	QCM-D sensing
q�X   E11 Action:T18q�X    T47	Modifier 431 440	adsorbed 
q�X!   T48	Reagent 460 472	HBS baseline
q�X   R17 Mod-Link Arg1:T46 Arg2:T47q�X   T27	Modifier 635 644	saturated
q�X   R5 Mod-Link Arg1:T31 Arg2:T27q�eX   statusq�X   linksq�]q�(c__main__
Link
q�(X   E1q�X   Productq�c__main__
Tag
q�(X   T1q�X   Actionq�KXK_]q�X   Prepareq�atq�q�h�(X   T9q�X   Reagentq�KbKw]q�(X   Hepesq�X   Bufferedq�X   Salineq�etq��q�tq��q�h�(h�X   Acts-onq�h�h�(X   T11q�X   Reagentq�K�K�]q�X   NaClq�atq��q�tr   �r  h�(h�X   Acts-on2r  h�h�(X   T13r  X   Reagentr  K�K�]r  X   Hepesr  atr  �r  tr	  �r
  h�(X   E2r  X   Productr  h�(X   T2r  X   Actionr  K�K�]r  X   preparer  atr  �r  h�(X   T16r  X   Reagentr  K�K�]r  X   streptavidinr  atr  �r  tr  �r  h�(X   E3r  X   Acts-onr  h�(X   T3r  X   Actionr  K�M]r  (X   maker   X   ar!  X   baseliner"  etr#  �r$  h�(X   T14r%  X   Reagentr&  MM ]r'  (X   HBSr(  X   solutionr)  etr*  �r+  tr,  �r-  h�(X   E4r.  X   Acts-onr/  h�(X   T4r0  X   Actionr1  M"M(]r2  X   Exposer3  atr4  �r5  h�(X   T20r6  X   Devicer7  M6M=]r8  X   sensorsr9  atr:  �r;  tr<  �r=  h�(j.  X   Siter>  j5  h�(X   T22r?  X   Reagentr@  MMM_]rA  (X   FetalrB  X   BovinerC  X   SerumrD  etrE  �rF  trG  �rH  h�(j.  X   UsingrI  j5  h�(X   T24rJ  X   MethodrK  MzM�]rL  (X   staticrM  X
   conditionsrN  etrO  �rP  trQ  �rR  h�(j.  X   Using2rS  j5  h�(X   T25rT  X   DevicerU  M�M�]rV  X   QCM-DrW  atrX  �rY  trZ  �r[  h�(X   E6r\  X   Acts-onr]  h�(X   T6r^  X   Actionr_  M�M�]r`  X   Rinsera  atrb  �rc  h�(X   T29rd  X   Devicere  MM	]rf  X   sensorsrg  atrh  �ri  trj  �rk  h�(j\  X   Usingrl  jc  h�(X   T30rm  X   Reagentrn  MM]ro  (X   HBSrp  X   solutionrq  etrr  �rs  trt  �ru  h�(X   E7rv  X   Acts-onrw  h�(X   T7rx  X   Actionry  M!M%]rz  X   Flowr{  atr|  �r}  h�(X   T31r~  X   Reagentr  M*M?]r�  (X   streptavidinr�  X   solutionr�  etr�  �r�  tr�  �r�  h�(jv  X   Siter�  j}  h�(X   T33r�  X   Devicer�  MRMY]r�  X   sensorsr�  atr�  �r�  tr�  �r�  h�(X   E8r�  X   Acts-onr�  h�(X   T8r�  X   Actionr�  M�M�]r�  X   Rinser�  atr�  �r�  h�(X   T35r�  X   Devicer�  M�M�]r�  X   sensorsr�  atr�  �r�  tr�  �r�  h�(j�  X   Usingr�  j�  h�(X   T36r�  X   Reagentr�  M�M�]r�  (X   HBSr�  X   solutionr�  etr�  �r�  tr�  �r�  h�(X   R1r�  X   Measurer�  j  h�(X   T12r�  X   Concentrationr�  K�K�]r�  (X   10r�  X   mMr�  etr�  �r�  tr�  �r�  h�(X   R2r�  X   Measurer�  h�h�(X   T10r�  X   Concentrationr�  K�K�]r�  (X   150r�  X   mMr�  etr�  �r�  tr�  �r�  h�(X   R3r�  X   Measurer�  j  h�(X   T15r�  X   Concentrationr�  K�K�]r�  (X   25r�  X   µgr�  X   /r�  X   mlr�  etr�  �r�  tr�  �r�  h�(X   R4r�  X   Meronymr�  j  h�(X   T17r�  X   Reagentr�  K�K�]r�  X   HBSr�  atr�  �r�  tr�  �r�  h�(X   R7r�  X   Mod-Linkr�  jF  h�(X   T21r�  X   Modifierr�  MAML]r�  X   non-dilutedr�  atr�  �r�  tr�  �r�  h�(X   R8r�  X   Mod-Linkr�  j;  h�(X   T19r�  X   Modifierr�  M-M5]r�  X   modifiedr�  atr�  �r�  tr�  �r�  h�(X   R9r�  X   Settingr�  j5  h�(X   T23r�  X   Timer�  MjMs]r�  (X   30-60r�  X   minr�  etr�  �r�  tr�  �r�  h�(X   R10r�  X   Mod-Linkr�  j�  h�(X   T32r�  X   Modifierr�  MIMQ]r�  X   modifiedr�  atr�  �r�  tr�  �r�  h�(X   R11r   X   Measurer  j�  h�(X   T34r  X   Generic-Measurer  M[Mi]r  (X   100-500r  X   µlr  j�  X   minr  etr  �r	  tr
  �r  h�(X   E9r  X   Acts-onr  h�(X   T37r  X   Actionr  K K]r  X   Investigatingr  atr  �r  h�(X   T39r  X   Reagentr  KK*]r  (X   proteinr  X   bindingr  etr  �r  tr  �r  h�(j  X   Acts-on2r  j  h�(X   T40r  X   Reagentr  K/KC]r   (X   streptavidinr!  X   bindingr"  etr#  �r$  tr%  �r&  h�(X   R12r'  X   Coreference-Linkr(  h�(X   T42r)  X   Reagentr*  KyK|]r+  X   HBSr,  atr-  �r.  h�tr/  �r0  h�(X   R13r1  X   Mod-Linkr2  j  h�(X   T38r3  X   Modifierr4  KK]r5  X   non-specificr6  atr7  �r8  tr9  �r:  h�(X   R14r;  X   Meronymr<  h�h�tr=  �r>  h�(X   R15r?  X   Meronymr@  j  h�trA  �rB  h�(X   R16rC  X   Coreference-LinkrD  h�(X   T45rE  X   ReagentrF  MaMd]rG  X   FBSrH  atrI  �rJ  jF  trK  �rL  h�(X   R17rM  X   Mod-LinkrN  h�(X   T46rO  X   ReagentrP  M�M�]rQ  X   FBSrR  atrS  �rT  h�(X   T47rU  X   ModifierrV  M�M�]rW  X   adsorbedrX  atrY  �rZ  tr[  �r\  h�(X   R5r]  X   Mod-Linkr^  j�  h�(X   T27r_  X   Modifierr`  M{M�]ra  X	   saturatedrb  atrc  �rd  tre  �rf  eX   headingrg  ]rh  (X   Investigatingri  X   non-specificrj  X   proteinrk  X   bindingrl  X   andrm  X   streptavidinrn  X   bindingro  X   usingrp  X   QCM-Drq  X   sensingrr  eX   sentsrs  ]rt  (]ru  (X   Preparerv  j!  X   Hepesrw  X   Bufferedrx  X   Salinery  X   (rz  X   HBSr{  X   )r|  X   solutionr}  X
   containingr~  X   150r  X   mMr�  X   NaClr�  X   andr�  X   10r�  X   mMr�  X   Hepesr�  X   .r�  e]r�  (X   Thenr�  X   preparer�  j!  X   solutionr�  X   ofr�  X   25r�  X   µgr�  j�  X   mlr�  X   streptavidinr�  X   inr�  X   HBSr�  j�  e]r�  (X   Usingr�  X   QCM-Dr�  X   sensingr�  X   ,r�  X   maker�  j!  X   baseliner�  X   ofr�  X   ther�  X   HBSr�  X   solutionr�  j�  e]r�  (X   Exposer�  X   ther�  X   modifiedr�  X   sensorsr�  X   tor�  X   non-dilutedr�  X   Fetalr�  X   Boviner�  X   Serumr�  jz  X   FBSr�  j|  X   forr�  X   30-60r�  X   minr�  X   underr�  X   staticr�  X
   conditionsr�  X   inr�  X   QCM-Dr�  X   andr�  X   measurer�  X   ther�  X   amountr�  X   ofr�  X   adsorbedr�  X   FBSr�  X   relativer�  X   tor�  X   ther�  X   HBSr�  X   baseliner�  X   obtainedr�  X   inr�  X   ther�  X   previousr�  X   stepr�  j�  e]r�  (X   Rinser�  X   ther�  X   sensorsr�  X   withr�  X   ther�  X   HBSr�  X   solutionr�  j�  e]r�  (X   Flowr�  X   ther�  X   streptavidinr�  X   solutionr�  X   overr�  X   ther�  X   modifiedr�  X   sensorsr�  jz  X   100-500r�  X   µlr�  j�  X   minr�  j|  j�  X   untilr�  X   theyr�  X   arer�  X	   saturatedr�  j�  e]r�  (X   Rinser�  X   ther�  X   sensorsr�  X   withr�  X   ther�  X   HBSr�  X   solutionr�  j�  eeX   tagsr�  ]r�  (h�j  j$  j5  h�(X   T5r�  X   Actionr�  M�M�]r�  X   measurer�  atr�  �r�  jc  j}  j�  h�j�  h�j�  j  j�  j  j�  j+  j�  j;  j�  jF  j�  jP  jY  h�(X   T26r�  X   Amountr�  M�M�]r�  X   amountr�  atr�  �r�  ji  js  j�  j�  j�  j	  j�  j�  j  j8  j  j$  h�(X   T41r�  X   Actionr�  KJKW]r�  (X   QCM-Dr�  X   sensingr�  etr�  �r�  j.  h�(X   T43r�  X   Reagentr�  K~K�]r�  X   solutionr�  atr�  �r�  h�(X   T44r�  X   Reagentr�  K�K�]r�  X   solutionr�  atr   �r  jJ  jT  h�(X   T18r  X   Actionr  K�K�]r  (X   QCM-Dr  X   sensingr  etr  �r  jZ  h�(X   T48r	  X   Reagentr
  M�M�]r  (X   HBSr  X   baseliner  etr  �r  jd  eX   unique_tagsr  cbuiltins
set
r  ]r  (jK  j�  h�j�  j�  j�  j  h�j7  e�r  Rr  X   tag_0_idr  X   T0r  X
   tag_0_namer  h,X   tokens2dr  ]r  (]r  (c__main__
Token
r  )�r  }r  (X   wordr  h�X   labelr  X   B-Actionr   X   originalr!  h�X   feature_valuesr"  Nubj  )�r#  }r$  (j  j!  j  h,j!  j!  j"  Nubj  )�r%  }r&  (j  h�j  X	   B-Reagentr'  j!  h�j"  Nubj  )�r(  }r)  (j  h�j  X	   I-Reagentr*  j!  h�j"  Nubj  )�r+  }r,  (j  h�j  X	   I-Reagentr-  j!  h�j"  Nubj  )�r.  }r/  (j  jz  j  h,j!  jz  j"  Nubj  )�r0  }r1  (j  j,  j  X	   B-Reagentr2  j!  j,  j"  Nubj  )�r3  }r4  (j  j|  j  h,j!  j|  j"  Nubj  )�r5  }r6  (j  j�  j  X	   B-Reagentr7  j!  j�  j"  Nubj  )�r8  }r9  (j  j~  j  h,j!  j~  j"  Nubj  )�r:  }r;  (j  j�  j  X   B-Concentrationr<  j!  j�  j"  Nubj  )�r=  }r>  (j  j�  j  X   I-Concentrationr?  j!  j�  j"  Nubj  )�r@  }rA  (j  h�j  X	   B-ReagentrB  j!  h�j"  Nubj  )�rC  }rD  (j  j�  j  h,j!  j�  j"  Nubj  )�rE  }rF  (j  j�  j  X   B-ConcentrationrG  j!  j�  j"  Nubj  )�rH  }rI  (j  j�  j  X   I-ConcentrationrJ  j!  j�  j"  Nubj  )�rK  }rL  (j  j  j  X	   B-ReagentrM  j!  j  j"  Nubj  )�rN  }rO  (j  j�  j  h,j!  j�  j"  Nube]rP  (j  )�rQ  }rR  (j  j�  j  h,j!  j�  j"  Nubj  )�rS  }rT  (j  j  j  X   B-ActionrU  j!  j  j"  Nubj  )�rV  }rW  (j  j!  j  h,j!  j!  j"  Nubj  )�rX  }rY  (j  j�  j  X	   B-ReagentrZ  j!  j�  j"  Nubj  )�r[  }r\  (j  j�  j  h,j!  j�  j"  Nubj  )�r]  }r^  (j  j�  j  X   B-Concentrationr_  j!  j�  j"  Nubj  )�r`  }ra  (j  j�  j  X   I-Concentrationrb  j!  j�  j"  Nubj  )�rc  }rd  (j  j�  j  X   I-Concentrationre  j!  j�  j"  Nubj  )�rf  }rg  (j  j�  j  X   I-Concentrationrh  j!  j�  j"  Nubj  )�ri  }rj  (j  j  j  X	   B-Reagentrk  j!  j  j"  Nubj  )�rl  }rm  (j  j�  j  h,j!  j�  j"  Nubj  )�rn  }ro  (j  j�  j  X	   B-Reagentrp  j!  j�  j"  Nubj  )�rq  }rr  (j  j�  j  h,j!  j�  j"  Nube]rs  (j  )�rt  }ru  (j  j�  j  h,j!  j�  j"  Nubj  )�rv  }rw  (j  j  j  X   B-Actionrx  j!  j  j"  Nubj  )�ry  }rz  (j  j  j  X   I-Actionr{  j!  j  j"  Nubj  )�r|  }r}  (j  j�  j  h,j!  j�  j"  Nubj  )�r~  }r  (j  j   j  X   B-Actionr�  j!  j   j"  Nubj  )�r�  }r�  (j  j!  j  X   I-Actionr�  j!  j!  j"  Nubj  )�r�  }r�  (j  j"  j  X   I-Actionr�  j!  j"  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j(  j  X	   B-Reagentr�  j!  j(  j"  Nubj  )�r�  }r�  (j  j)  j  X	   I-Reagentr�  j!  j)  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nube]r�  (j  )�r�  }r�  (j  j3  j  X   B-Actionr�  j!  j3  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X
   B-Modifierr�  j!  j�  j"  Nubj  )�r�  }r�  (j  j9  j  X   B-Devicer�  j!  j9  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X
   B-Modifierr�  j!  j�  j"  Nubj  )�r�  }r�  (j  jB  j  X	   B-Reagentr�  j!  jB  j"  Nubj  )�r�  }r�  (j  jC  j  X	   I-Reagentr�  j!  jC  j"  Nubj  )�r�  }r�  (j  jD  j  X	   I-Reagentr�  j!  jD  j"  Nubj  )�r�  }r�  (j  jz  j  h,j!  jz  j"  Nubj  )�r�  }r�  (j  jH  j  X	   B-Reagentr�  j!  jH  j"  Nubj  )�r�  }r�  (j  j|  j  h,j!  j|  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X   B-Timer�  j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X   I-Timer�  j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  jM  j  X   B-Methodr�  j!  jM  j"  Nubj  )�r�  }r�  (j  jN  j  X   I-Methodr�  j!  jN  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  jW  j  X   B-Devicer�  j!  jW  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X   B-Actionr�  j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  X   B-Amountr�  j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  jX  j  X
   B-Modifierr�  j!  jX  j"  Nubj  )�r�  }r�  (j  jR  j  X	   B-Reagentr�  j!  jR  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j  j  X	   B-Reagentr�  j!  j  j"  Nubj  )�r�  }r�  (j  j  j  X	   I-Reagentr�  j!  j  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nube]r�  (j  )�r�  }r�  (j  ja  j  X   B-Actionr�  j!  ja  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  jg  j  X   B-Devicer�  j!  jg  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r�  }r�  (j  j�  j  h,j!  j�  j"  Nubj  )�r   }r  (j  jp  j  X	   B-Reagentr  j!  jp  j"  Nubj  )�r  }r  (j  jq  j  X	   I-Reagentr  j!  jq  j"  Nubj  )�r  }r  (j  j�  j  h,j!  j�  j"  Nube]r  (j  )�r	  }r
  (j  j{  j  X   B-Actionr  j!  j{  j"  Nubj  )�r  }r  (j  j�  j  h,j!  j�  j"  Nubj  )�r  }r  (j  j�  j  X	   B-Reagentr  j!  j�  j"  Nubj  )�r  }r  (j  j�  j  X	   I-Reagentr  j!  j�  j"  Nubj  )�r  }r  (j  j�  j  h,j!  j�  j"  Nubj  )�r  }r  (j  j�  j  h,j!  j�  j"  Nubj  )�r  }r  (j  j�  j  X
   B-Modifierr  j!  j�  j"  Nubj  )�r  }r  (j  j�  j  X   B-Devicer  j!  j�  j"  Nubj  )�r  }r  (j  jz  j  h,j!  jz  j"  Nubj  )�r   }r!  (j  j  j  X   B-Generic-Measurer"  j!  j  j"  Nubj  )�r#  }r$  (j  j  j  X   I-Generic-Measurer%  j!  j  j"  Nubj  )�r&  }r'  (j  j�  j  X   I-Generic-Measurer(  j!  j�  j"  Nubj  )�r)  }r*  (j  j  j  X   I-Generic-Measurer+  j!  j  j"  Nubj  )�r,  }r-  (j  j|  j  h,j!  j|  j"  Nubj  )�r.  }r/  (j  j�  j  h,j!  j�  j"  Nubj  )�r0  }r1  (j  j�  j  h,j!  j�  j"  Nubj  )�r2  }r3  (j  j�  j  h,j!  j�  j"  Nubj  )�r4  }r5  (j  j�  j  h,j!  j�  j"  Nubj  )�r6  }r7  (j  jb  j  X
   B-Modifierr8  j!  jb  j"  Nubj  )�r9  }r:  (j  j�  j  h,j!  j�  j"  Nube]r;  (j  )�r<  }r=  (j  j�  j  X   B-Actionr>  j!  j�  j"  Nubj  )�r?  }r@  (j  j�  j  h,j!  j�  j"  Nubj  )�rA  }rB  (j  j�  j  X   B-DevicerC  j!  j�  j"  Nubj  )�rD  }rE  (j  j�  j  h,j!  j�  j"  Nubj  )�rF  }rG  (j  j�  j  h,j!  j�  j"  Nubj  )�rH  }rI  (j  j�  j  X	   B-ReagentrJ  j!  j�  j"  Nubj  )�rK  }rL  (j  j�  j  X	   I-ReagentrM  j!  j�  j"  Nubj  )�rN  }rO  (j  j�  j  h,j!  j�  j"  NubeeX   word_cntrP  KuX   f_dfrQ  NX   pos_tagsrR  ]rS  (]rT  (X   PreparerU  X   VBrV  X   B-VPrW  �rX  j!  X   DTrY  X   B-NPrZ  �r[  X   Hepesr\  X   NNPr]  X   I-NPr^  �r_  X   Bufferedr`  X   NNPra  X   I-NPrb  �rc  X   Salinerd  X   NNPre  X   I-NPrf  �rg  jz  jz  h,�rh  X   HBSri  X   NNrj  X   B-NPrk  �rl  j|  j|  h,�rm  X   solutionrn  X   NNro  X   B-NPrp  �rq  X
   containingrr  X   VBGrs  X   B-VPrt  �ru  X   150rv  X   CDrw  X   B-NPrx  �ry  X   mMrz  X   NNr{  X   I-NPr|  �r}  X   NaClr~  X   NNr  X   I-NPr�  �r�  X   andr�  X   CCr�  h,�r�  X   10r�  X   CDr�  X   B-NPr�  �r�  X   mMr�  X   NNr�  X   I-NPr�  �r�  X   Hepesr�  X   NNPr�  X   I-NPr�  �r�  j�  j�  h,�r�  e]r�  (X   Thenr�  X   RBr�  h,�r�  X   preparer�  X   VBr�  X   B-VPr�  �r�  j!  X   DTr�  X   B-NPr�  �r�  X   solutionr�  X   NNr�  X   I-NPr�  �r�  X   ofr�  X   INr�  X   B-PPr�  �r�  X   25r�  X   CDr�  X   B-NPr�  �r�  X   µgr�  X   NNr�  X   I-NPr�  �r�  j�  X   SYMr�  X   B-NPr�  �r�  X   mlr�  X   NNr�  X   I-NPr�  �r�  X   streptavidinr�  X   NNr�  X   I-NPr�  �r�  X   inr�  X   INr�  X   B-PPr�  �r�  X   HBSr�  X   NNPr�  X   B-NPr�  �r�  j�  j�  h,�r�  e]r�  (X   Usingr�  X   VBGr�  X   B-VPr�  �r�  X   QCM-Dr�  X   NNr�  X   B-NPr�  �r�  X   sensingr�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  X   maker�  X   VBPr�  X   B-VPr�  �r�  j!  X   DTr�  X   B-NPr�  �r�  X   baseliner�  X   NNr�  X   I-NPr�  �r�  X   ofr�  X   INr�  X   B-PPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   HBSr�  X   NNPr�  X   I-NPr�  �r�  X   solutionr�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  e]r�  (X   Exposer�  X   VBr�  X   B-VPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   modifiedr�  X   VBNr�  X   I-NPr�  �r�  X   sensorsr�  X   NNSr�  X   I-NPr�  �r�  X   tor�  X   TOr�  X   B-PPr�  �r�  X   non-dilutedr   X   JJr  X   B-NPr  �r  X   Fetalr  X   JJr  X   I-NPr  �r  X   Boviner  X   JJr	  X   I-NPr
  �r  X   Serumr  X   NNr  X   I-NPr  �r  jz  jz  h,�r  X   FBSr  X   NNr  X   B-NPr  �r  j|  j|  h,�r  X   forr  X   INr  X   B-PPr  �r  X   30-60r  X   CDr  X   B-NPr  �r  X   minr  X   NNr  X   I-NPr   �r!  X   underr"  X   INr#  X   B-PPr$  �r%  X   staticr&  X   JJr'  X   B-NPr(  �r)  X
   conditionsr*  X   NNSr+  X   I-NPr,  �r-  X   inr.  X   INr/  X   B-PPr0  �r1  X   QCM-Dr2  X   NNr3  X   B-NPr4  �r5  X   andr6  X   CCr7  h,�r8  X   measurer9  X   VBr:  X   B-VPr;  �r<  X   ther=  X   DTr>  X   B-NPr?  �r@  X   amountrA  X   NNrB  X   I-NPrC  �rD  X   ofrE  X   INrF  X   B-PPrG  �rH  X   adsorbedrI  X   VBNrJ  X   B-NPrK  �rL  X   FBSrM  X   NNrN  X   I-NPrO  �rP  X   relativerQ  X   JJrR  X   B-ADVPrS  �rT  X   torU  X   TOrV  X   B-PPrW  �rX  X   therY  X   DTrZ  X   B-NPr[  �r\  X   HBSr]  X   NNr^  X   I-NPr_  �r`  X   baselinera  X   NNrb  X   I-NPrc  �rd  X   obtainedre  X   VBNrf  X   B-VPrg  �rh  X   inri  X   INrj  X   B-PPrk  �rl  X   therm  X   DTrn  X   B-NPro  �rp  X   previousrq  X   JJrr  X   I-NPrs  �rt  X   stepru  X   NNrv  X   I-NPrw  �rx  j�  j�  h,�ry  e]rz  (X   Rinser{  X   VBr|  X   B-VPr}  �r~  X   ther  X   DTr�  X   B-NPr�  �r�  X   sensorsr�  X   NNSr�  X   I-NPr�  �r�  X   withr�  X   INr�  X   B-PPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   HBSr�  X   NNPr�  X   I-NPr�  �r�  X   solutionr�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  e]r�  (X   Flowr�  X   VBr�  X   B-VPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   streptavidinr�  X   NNr�  X   I-NPr�  �r�  X   solutionr�  X   NNr�  X   I-NPr�  �r�  X   overr�  X   INr�  X   B-PPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   modifiedr�  X   VBNr�  X   I-NPr�  �r�  X   sensorsr�  X   NNSr�  X   I-NPr�  �r�  jz  jz  h,�r�  X   100-500r�  X   CDr�  X   B-NPr�  �r�  X   µlr�  X   NNr�  X   I-NPr�  �r�  j�  X   SYMr�  X   B-NPr�  �r�  X   minr�  X   NNr�  X   I-NPr�  �r�  j|  j|  h,�r�  j�  j�  h,�r�  X   untilr�  X   INr�  X   B-SBARr�  �r�  X   theyr�  X   PRPr�  X   B-NPr�  �r�  X   arer�  X   VBPr�  X   B-VPr�  �r�  X	   saturatedr�  X   VBNr�  X   I-VPr�  �r�  j�  j�  h,�r�  e]r�  (X   Rinser�  X   VBr�  X   B-VPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   sensorsr�  X   NNSr�  X   I-NPr�  �r�  X   withr�  X   INr�  X   B-PPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   HBSr�  X   NNPr�  X   I-NPr�  �r�  X   solutionr�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  eeX
   conll_depsr�  ]r�  (X  1	Prepare/VB	_	B	B	_	2	dep	_	_
2	a/DT	_	B	B	_	0	root	_	_
3	Hepes/NNP	_	I	I	_	2	dep	_	_
4	Buffered/NNP	_	I	I	_	3	dep	_	_
5	Saline/NNP	_	I	I	_	3	dep	_	_
6	(/(	_	O	O	_	3	dep	_	_
7	HBS/NN	_	B	B	_	3	dep	_	_
8	)/)	_	O	O	_	3	dep	_	_
9	solution/NN	_	B	B	_	3	dep	_	_
10	containing/VBG	_	B	B	_	3	dep	_	_
11	150/CD	_	B	B	_	3	dep	_	_
12	mM/NN	_	I	I	_	3	dep	_	_
13	NaCl/NN	_	I	I	_	3	dep	_	_
14	and/CC	_	O	O	_	3	dep	_	_
15	10/CD	_	B	B	_	3	dep	_	_
16	mM/NN	_	I	I	_	3	dep	_	_
17	Hepes/NNP	_	I	I	_	3	dep	_	_
18	./.	_	O	O	_	2	dep	_	_
r�  Xq  1	Then/RB	_	O	O	_	2	dep	_	_
2	prepare/VB	_	B	B	_	0	root	_	_
3	a/DT	_	B	B	_	2	dep	_	_
4	solution/NN	_	I	I	_	3	dep	_	_
5	of/IN	_	B	B	_	3	dep	_	_
6	25/CD	_	B	B	_	3	dep	_	_
7	µg/NN	_	I	I	_	9	dep	_	_
8	//SYM	_	B	B	_	9	dep	_	_
9	ml/NN	_	I	I	_	3	nummod	_	_
10	streptavidin/NN	_	I	I	_	3	dep	_	_
11	in/IN	_	B	B	_	3	dep	_	_
12	HBS/NNP	_	B	B	_	3	dep	_	_
13	./.	_	O	O	_	2	dep	_	_
r�  XX  1	Using/VBG	_	B	B	_	5	nsubj	_	_
2	QCM-D/NN	_	B	B	_	1	dep	_	_
3	sensing/NN	_	I	I	_	5	dep	_	_
4	,/,	_	O	O	_	5	dep	_	_
5	make/VBP	_	B	B	_	0	root	_	_
6	a/DT	_	B	B	_	5	dep	_	_
7	baseline/NN	_	I	I	_	5	dep	_	_
8	of/IN	_	B	B	_	9	dep	_	_
9	the/DT	_	B	B	_	5	nmod	_	_
10	HBS/NNP	_	I	I	_	9	dep	_	_
11	solution/NN	_	I	I	_	9	dep	_	_
12	./.	_	O	O	_	5	dep	_	_
r�  Xf  1	Expose/VB	_	B	B	_	2	advmod	_	_
2	the/DT	_	B	B	_	0	root	_	_
3	modified/VBN	_	I	I	_	2	dep	_	_
4	sensors/NNS	_	I	I	_	3	dep	_	_
5	to/TO	_	B	B	_	3	dep	_	_
6	non-diluted/JJ	_	B	B	_	3	dep	_	_
7	Fetal/JJ	_	I	I	_	3	dep	_	_
8	Bovine/JJ	_	I	I	_	3	dep	_	_
9	Serum/NN	_	I	I	_	3	dep	_	_
10	(/(	_	O	O	_	3	dep	_	_
11	FBS/NN	_	B	B	_	3	dep	_	_
12	)/)	_	O	O	_	3	dep	_	_
13	for/IN	_	B	B	_	3	dep	_	_
14	30-60/CD	_	B	B	_	3	dep	_	_
15	min/NN	_	I	I	_	3	dep	_	_
16	under/IN	_	B	B	_	3	dep	_	_
17	static/JJ	_	B	B	_	3	dep	_	_
18	conditions/NNS	_	I	I	_	3	dep	_	_
19	in/IN	_	B	B	_	3	dep	_	_
20	QCM-D/NN	_	B	B	_	3	dep	_	_
21	and/CC	_	O	O	_	3	dep	_	_
22	measure/VB	_	B	B	_	3	dep	_	_
23	the/DT	_	B	B	_	3	dep	_	_
24	amount/NN	_	I	I	_	3	dep	_	_
25	of/IN	_	B	B	_	3	dep	_	_
26	adsorbed/VBN	_	B	B	_	3	dep	_	_
27	FBS/NN	_	I	I	_	3	dep	_	_
28	relative/JJ	_	B	B	_	3	dep	_	_
29	to/TO	_	B	B	_	3	dep	_	_
30	the/DT	_	B	B	_	3	dep	_	_
31	HBS/NN	_	I	I	_	3	dep	_	_
32	baseline/NN	_	I	I	_	3	dep	_	_
33	obtained/VBN	_	B	B	_	3	dep	_	_
34	in/IN	_	B	B	_	3	dep	_	_
35	the/DT	_	B	B	_	3	dep	_	_
36	previous/JJ	_	I	I	_	3	dep	_	_
37	step/NN	_	I	I	_	3	dep	_	_
38	./.	_	O	O	_	2	dep	_	_
r�  X�   1	Rinse/VB	_	B	B	_	5	nsubj	_	_
2	the/DT	_	B	B	_	1	dep	_	_
3	sensors/NNS	_	I	I	_	5	dep	_	_
4	with/IN	_	B	B	_	5	dep	_	_
5	the/DT	_	B	B	_	0	root	_	_
6	HBS/NNP	_	I	I	_	5	dep	_	_
7	solution/NN	_	I	I	_	5	dep	_	_
8	./.	_	O	O	_	5	dep	_	_
r   XF  1	Flow/VB	_	B	B	_	2	dep	_	_
2	the/DT	_	B	B	_	0	root	_	_
3	streptavidin/NN	_	I	I	_	2	dep	_	_
4	solution/NN	_	I	I	_	3	dep	_	_
5	over/IN	_	B	B	_	3	dep	_	_
6	the/DT	_	B	B	_	3	dep	_	_
7	modified/VBN	_	I	I	_	3	dep	_	_
8	sensors/NNS	_	I	I	_	3	dep	_	_
9	(/(	_	O	O	_	3	dep	_	_
10	100-500/CD	_	B	B	_	3	dep	_	_
11	µl/NN	_	I	I	_	3	dep	_	_
12	//SYM	_	B	B	_	3	dep	_	_
13	min/NN	_	I	I	_	3	dep	_	_
14	)/)	_	O	O	_	3	dep	_	_
15	,/,	_	O	O	_	3	dep	_	_
16	until/IN	_	B	B	_	3	dep	_	_
17	they/PRP	_	B	B	_	3	dep	_	_
18	are/VBP	_	B	B	_	3	dep	_	_
19	saturated/VBN	_	I	I	_	3	dep	_	_
20	./.	_	O	O	_	2	dep	_	_
r  X�   1	Rinse/VB	_	B	B	_	5	nsubj	_	_
2	the/DT	_	B	B	_	1	dep	_	_
3	sensors/NNS	_	I	I	_	5	dep	_	_
4	with/IN	_	B	B	_	5	dep	_	_
5	the/DT	_	B	B	_	0	root	_	_
6	HBS/NNP	_	I	I	_	5	dep	_	_
7	solution/NN	_	I	I	_	5	dep	_	_
8	./.	_	O	O	_	5	dep	_	_
r  eX   parse_treesr  ]r  (cnltk.tree
Tree
r  )�r  j  )�r  (j  )�r  (j  )�r	  X   Preparer
  a}r  X   _labelr  X   VBr  sbj  )�r  (j  )�r  (j  )�r  j!  a}r  j  X   DTr  sbj  )�r  (j  )�r  (j  )�r  X   Hepesr  a}r  j  X   NNPr  sbj  )�r  X   Bufferedr  a}r  j  X   NNPr  sbj  )�r  X   Saliner  a}r  j  X   NNPr   sbe}r!  j  X   NPr"  sbj  )�r#  (j  )�r$  X   -LRB-r%  a}r&  j  X   -LRB-r'  sbj  )�r(  j  )�r)  X   HBSr*  a}r+  j  X   NNPr,  sba}r-  j  X   NPr.  sbj  )�r/  X   -RRB-r0  a}r1  j  X   -RRB-r2  sbe}r3  j  X   PRNr4  sbe}r5  j  X   NPr6  sbj  )�r7  X   solutionr8  a}r9  j  X   NNr:  sbe}r;  j  X   NPr<  sbj  )�r=  (j  )�r>  X
   containingr?  a}r@  j  X   VBGrA  sbj  )�rB  (j  )�rC  (j  )�rD  X   150rE  a}rF  j  X   CDrG  sbj  )�rH  X   mMrI  a}rJ  j  X   NNrK  sbj  )�rL  X   NaClrM  a}rN  j  X   NNPrO  sbe}rP  j  X   NPrQ  sbj  )�rR  X   andrS  a}rT  j  X   CCrU  sbj  )�rV  (j  )�rW  X   10rX  a}rY  j  X   CDrZ  sbj  )�r[  X   mMr\  a}r]  j  X   NNr^  sbj  )�r_  X   Hepesr`  a}ra  j  X   NNSrb  sbe}rc  j  X   NPrd  sbe}re  j  X   NPrf  sbe}rg  j  X   VPrh  sbe}ri  j  X   NPrj  sbe}rk  j  X   VPrl  sbj  )�rm  j�  a}rn  j  j�  sbe}ro  j  h0sba}rp  j  X   ROOTrq  sbj  )�rr  j  )�rs  (j  )�rt  (j  )�ru  X   Thenrv  a}rw  j  X   RBrx  sbj  )�ry  (j  )�rz  X   preparer{  a}r|  j  X   VBGr}  sbj  )�r~  (j  )�r  (j  )�r�  j!  a}r�  j  X   DTr�  sbj  )�r�  X   solutionr�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   ofr�  a}r�  j  X   INr�  sbj  )�r�  (j  )�r�  X   25r�  a}r�  j  X   CDr�  sbj  )�r�  X   µgr�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   VPr�  sbe}r�  j  h0sbj  )�r�  j�  a}r�  j  X   :r�  sbj  )�r�  (j  )�r�  j  )�r�  X   mlr�  a}r�  j  X   NNr�  sba}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   streptavidinr�  a}r�  j  X   VBNr�  sbj  )�r�  (j  )�r�  X   inr�  a}r�  j  X   INr�  sbj  )�r�  j  )�r�  X   HBSr�  a}r�  j  X   NNPr�  sba}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   VPr�  sbe}r�  j  X   NPr�  sbj  )�r�  j�  a}r�  j  j�  sbe}r�  j  X   FRAGr�  sba}r�  j  X   ROOTr�  sbj  )�r�  j  )�r�  (j  )�r�  j  )�r�  (j  )�r�  X   Usingr�  a}r�  j  X   VBGr�  sbj  )�r�  (j  )�r�  X   QCM-Dr�  a}r�  j  X   NNPr�  sbj  )�r�  X   sensingr�  a}r�  j  X   NNPr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   VPr�  sba}r�  j  h0sbj  )�r�  j�  a}r�  j  j�  sbj  )�r�  (j  )�r�  X   maker�  a}r�  j  X   VBPr�  sbj  )�r�  (j  )�r�  (j  )�r�  j!  a}r�  j  X   DTr�  sbj  )�r�  X   baseliner�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   ofr�  a}r�  j  X   INr�  sbj  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   HBSr�  a}r�  j  X   NNPr�  sbj  )�r�  X   solutionr�  a}r   j  X   NNr  sbe}r  j  X   NPr  sbe}r  j  X   PPr  sbe}r  j  X   NPr  sbe}r  j  X   VPr	  sbj  )�r
  j�  a}r  j  j�  sbe}r  j  h0sba}r  j  X   ROOTr  sbj  )�r  j  )�r  (j  )�r  (j  )�r  (j  )�r  X   Exposer  a}r  j  X   VBr  sbj  )�r  (j  )�r  X   ther  a}r  j  X   DTr  sbj  )�r  X   modifiedr  a}r  j  X   VBNr  sbj  )�r   X   sensorsr!  a}r"  j  X   NNSr#  sbe}r$  j  X   NPr%  sbj  )�r&  (j  )�r'  X   tor(  a}r)  j  X   TOr*  sbj  )�r+  (j  )�r,  (j  )�r-  (j  )�r.  X   non-dilutedr/  a}r0  j  X   JJr1  sbj  )�r2  X   Fetalr3  a}r4  j  X   NNPr5  sbj  )�r6  X   Boviner7  a}r8  j  X   NNPr9  sbj  )�r:  X   Serumr;  a}r<  j  X   NNPr=  sbe}r>  j  X   NPr?  sbj  )�r@  (j  )�rA  X   -LRB-rB  a}rC  j  X   -LRB-rD  sbj  )�rE  j  )�rF  X   FBSrG  a}rH  j  X   NNPrI  sba}rJ  j  X   NPrK  sbj  )�rL  X   -RRB-rM  a}rN  j  X   -RRB-rO  sbe}rP  j  X   PRNrQ  sbe}rR  j  X   NPrS  sbj  )�rT  (j  )�rU  X   forrV  a}rW  j  X   INrX  sbj  )�rY  (j  )�rZ  X   30-60r[  a}r\  j  X   CDr]  sbj  )�r^  X   minr_  a}r`  j  X   NNra  sbe}rb  j  X   NPrc  sbe}rd  j  X   PPre  sbe}rf  j  X   NPrg  sbe}rh  j  X   PPri  sbj  )�rj  (j  )�rk  X   underrl  a}rm  j  X   INrn  sbj  )�ro  (j  )�rp  (j  )�rq  X   staticrr  a}rs  j  X   JJrt  sbj  )�ru  X
   conditionsrv  a}rw  j  X   NNSrx  sbe}ry  j  X   NPrz  sbj  )�r{  (j  )�r|  X   inr}  a}r~  j  X   INr  sbj  )�r�  j  )�r�  X   QCM-Dr�  a}r�  j  X   NNPr�  sba}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   VPr�  sbj  )�r�  X   andr�  a}r�  j  X   CCr�  sbj  )�r�  (j  )�r�  X   measurer�  a}r�  j  X   VBr�  sbj  )�r�  (j  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   amountr�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   ofr�  a}r�  j  X   INr�  sbj  )�r�  (j  )�r�  X   adsorbedr�  a}r�  j  X   JJr�  sbj  )�r�  X   FBSr�  a}r�  j  X   NNPr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   relativer�  a}r�  j  X   JJr�  sbj  )�r�  (j  )�r�  X   tor�  a}r�  j  X   TOr�  sbj  )�r�  (j  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   HBSr�  a}r�  j  X   NNPr�  sbj  )�r�  X   baseliner�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   obtainedr�  a}r�  j  X   VBNr�  sbj  )�r�  (j  )�r�  X   inr�  a}r�  j  X   INr�  sbj  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   previousr�  a}r�  j  X   JJr�  sbj  )�r�  X   stepr�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   VPr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   ADVPr�  sbe}r�  j  X   VPr�  sbe}r�  j  X   VPr�  sbj  )�r�  j�  a}r�  j  j�  sbe}r�  j  h0sba}r�  j  X   ROOTr�  sbj  )�r�  j  )�r�  (j  )�r   (j  )�r  X   Rinser  a}r  j  X   VBr  sbj  )�r  (j  )�r  X   ther  a}r  j  X   DTr	  sbj  )�r
  X   sensorsr  a}r  j  X   NNSr  sbe}r  j  X   NPr  sbj  )�r  (j  )�r  X   withr  a}r  j  X   INr  sbj  )�r  (j  )�r  X   ther  a}r  j  X   DTr  sbj  )�r  X   HBSr  a}r  j  X   NNPr  sbj  )�r  X   solutionr  a}r   j  X   NNr!  sbe}r"  j  X   NPr#  sbe}r$  j  X   PPr%  sbe}r&  j  X   VPr'  sbj  )�r(  j�  a}r)  j  j�  sbe}r*  j  h0sba}r+  j  X   ROOTr,  sbj  )�r-  j  )�r.  (j  )�r/  (j  )�r0  X   Flowr1  a}r2  j  X   VBr3  sbj  )�r4  (j  )�r5  X   ther6  a}r7  j  X   DTr8  sbj  )�r9  X   streptavidinr:  a}r;  j  X   NNr<  sbj  )�r=  X   solutionr>  a}r?  j  X   NNr@  sbe}rA  j  X   NPrB  sbj  )�rC  (j  )�rD  X   overrE  a}rF  j  X   INrG  sbj  )�rH  (j  )�rI  (j  )�rJ  (j  )�rK  X   therL  a}rM  j  X   DTrN  sbj  )�rO  X   modifiedrP  a}rQ  j  X   VBNrR  sbj  )�rS  X   sensorsrT  a}rU  j  X   NNSrV  sbe}rW  j  X   NPrX  sbj  )�rY  (j  )�rZ  X   -LRB-r[  a}r\  j  X   -LRB-r]  sbj  )�r^  (j  )�r_  X   100-500r`  a}ra  j  X   JJrb  sbj  )�rc  X   µlrd  a}re  j  X   NNrf  sbj  )�rg  j�  a}rh  j  X   NNri  sbj  )�rj  X   minrk  a}rl  j  X   NNrm  sbe}rn  j  X   NPro  sbj  )�rp  X   -RRB-rq  a}rr  j  X   -RRB-rs  sbe}rt  j  X   PRNru  sbe}rv  j  X   NPrw  sbj  )�rx  j�  a}ry  j  j�  sbj  )�rz  (j  )�r{  X   untilr|  a}r}  j  X   INr~  sbj  )�r  (j  )�r�  j  )�r�  X   theyr�  a}r�  j  X   PRPr�  sba}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   arer�  a}r�  j  X   VBPr�  sbj  )�r�  j  )�r�  X	   saturatedr�  a}r�  j  X   VBNr�  sba}r�  j  X   VPr�  sbe}r�  j  X   VPr�  sbe}r�  j  h0sbe}r�  j  X   SBARr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   VPr�  sbj  )�r�  j�  a}r�  j  j�  sbe}r�  j  h0sba}r�  j  X   ROOTr�  sbj  )�r�  j  )�r�  (j  )�r�  (j  )�r�  X   Rinser�  a}r�  j  X   VBr�  sbj  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   sensorsr�  a}r�  j  X   NNSr�  sbe}r�  j  X   NPr�  sbj  )�r�  (j  )�r�  X   withr�  a}r�  j  X   INr�  sbj  )�r�  (j  )�r�  X   ther�  a}r�  j  X   DTr�  sbj  )�r�  X   HBSr�  a}r�  j  X   NNPr�  sbj  )�r�  X   solutionr�  a}r�  j  X   NNr�  sbe}r�  j  X   NPr�  sbe}r�  j  X   PPr�  sbe}r�  j  X   VPr�  sbj  )�r�  j�  a}r�  j  j�  sbe}r�  j  h0sba}r�  j  X   ROOTr�  sbeubj  h�X   arg1_tagr�  h�X   arg2_tagr�  h�X
   parse_treer�  j  j"  Nubh)�r�  }r�  (hK hK K�r�  hKK�r�  h	hj  h�j�  h�j�  h�j�  j  j"  Nubh)�r�  }r�  (hK hK K�r�  hKK�r�  h	hj  j  j�  h�j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK	K
�r�  h	hj  j  j�  j  j�  j  j�  jr  j"  Nubh)�r�  }r�  (hKhKK�r�  hK	K�r�  h	hj  j  j�  j$  j�  j+  j�  j�  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj  j/  j�  j5  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK	�r�  h	hj  j>  j�  j5  j�  jF  j�  j  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj  jI  j�  j5  j�  jP  j�  j  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj  jS  j�  j5  j�  jY  j�  j  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj  j]  j�  jc  j�  ji  j�  j�  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj  jl  j�  jc  j�  js  j�  j�  j"  Nubh)�r�  }r�  (hKhK K�r�  hKK�r 	  h	hj  jw  j�  j}  j�  j�  j�  j-  j"  Nubh)�r	  }r	  (hKhK K�r	  hKK�r	  h	hj  j�  j�  j}  j�  j�  j�  j-  j"  Nubh)�r	  }r	  (hKhK K�r	  hKK�r	  h	hj  j�  j�  j�  j�  j�  j�  j�  j"  Nubh)�r		  }r
	  (hKhK K�r	  hKK�r	  h	hj  j�  j�  j�  j�  j�  j�  j�  j"  Nubh)�r	  }r	  (hK hKK�r	  hKK�r	  h	hj  j�  j�  j  j�  j�  j�  j  j"  Nubh)�r	  }r	  (hK hKK�r	  hK
K�r	  h	hj  j�  j�  h�j�  j�  j�  j  j"  Nubh)�r	  }r	  (hKhK	K
�r	  hKK	�r	  h	hj  j�  j�  j  j�  j�  j�  jr  j"  Nubh)�r	  }r	  (hKhK	K
�r	  hKK�r	  h	hj  j�  j�  j  j�  j�  j�  jr  j"  Nubh)�r	  }r	  (hKhKK	�r	  hKK�r 	  h	hj  j�  j�  jF  j�  j�  j�  j  j"  Nubh)�r!	  }r"	  (hKhKK�r#	  hKK�r$	  h	hj  j�  j�  j;  j�  j�  j�  j  j"  Nubh)�r%	  }r&	  (hKhK K�r'	  hKK�r(	  h	hj  j�  j�  j5  j�  j�  j�  j  j"  Nubh)�r)	  }r*	  (hKhKK�r+	  hKK�r,	  h	hj  j�  j�  j�  j�  j�  j�  j-  j"  Nubh)�r-	  }r.	  (hKhKK�r/	  hK	K�r0	  h	hj  j  j�  j�  j�  j	  j�  j-  j"  Nubh)�r1	  }r2	  (hKhK K �r3	  hK K �r4	  h	hj  j  j�  j  j�  j  j�  j�  j"  Nubh)�r5	  }r6	  (hKhK K �r7	  hK K �r8	  h	hj  j  j�  j  j�  j$  j�  j�  j"  Nubh)�r9	  }r:	  (hK hKK�r;	  hKK�r<	  h	hj  j(  j�  j.  j�  h�j�  j  j"  Nubh)�r=	  }r>	  (hKhK K �r?	  hK K �r@	  h	hj  j2  j�  j  j�  j8  j�  j�  j"  Nubh)�rA	  }rB	  (hK hKK�rC	  hKK�rD	  h	hj  j<  j�  h�j�  h�j�  j  j"  Nubh)�rE	  }rF	  (hK hKK�rG	  hKK�rH	  h	hj  j@  j�  j  j�  h�j�  j  j"  Nubh)�rI	  }rJ	  (hKhK
K�rK	  hKK	�rL	  h	hj  jD  j�  jJ  j�  jF  j�  j  j"  Nubh)�rM	  }rN	  (hKhKK�rO	  hKK�rP	  h	hj  jN  j�  jT  j�  jZ  j�  j  j"  Nubh)�rQ	  }rR	  (hKhKK�rS	  hKK�rT	  h	hj  j^  j�  j�  j�  jd  j�  j-  j"  Nubh)�rU	  }rV	  (hK hK K�rW	  hK
K�rX	  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�rY	  }rZ	  (hK hK K�r[	  hKK�r\	  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r]	  }r^	  (hK hK K�r_	  hKK�r`	  h	hj  h,j�  h�j�  j.  j�  j  j"  Nubh)�ra	  }rb	  (hK hK K�rc	  hKK	�rd	  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�re	  }rf	  (hKhKK�rg	  hKK	�rh	  h	hj  h,j�  j  j�  j�  j�  jr  j"  Nubh)�ri	  }rj	  (hKhKK�rk	  hKK�rl	  h	hj  h,j�  j  j�  j�  j�  jr  j"  Nubh)�rm	  }rn	  (hKhKK�ro	  hKK�rp	  h	hj  h,j�  j  j�  j  j�  jr  j"  Nubh)�rq	  }rr	  (hKhKK�rs	  hKK�rt	  h	hj  h,j�  j$  j�  j  j�  j�  j"  Nubh)�ru	  }rv	  (hKhK K�rw	  hKK�rx	  h	hj  h,j�  j5  j�  j�  j�  j  j"  Nubh)�ry	  }rz	  (hKhK K�r{	  hKK�r|	  h	hj  h,j�  j5  j�  j�  j�  j  j"  Nubh)�r}	  }r~	  (hKhK K�r	  hKK�r�	  h	hj  h,j�  j5  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj  h,j�  j5  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK
K�r�	  h	hj  h,j�  j5  j�  jJ  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj  h,j�  j5  j�  jT  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj  h,j�  j5  j�  jZ  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK �r�	  h	hj  h,j�  j5  j�  j  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hK K�r�	  h	hj  h,j�  j�  j�  j5  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  j;  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK	�r�	  h	hj  h,j�  j�  j�  jF  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  jP  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  jY  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hK
K�r�	  h	hj  h,j�  j�  j�  jJ  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  jT  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj  h,j�  j�  j�  jZ  j�  j  j"  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK �r�	  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj  h,j�  j}  j�  j�  j�  j-  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK	K�r�	  h	hj  h,j�  j}  j�  j	  j�  j-  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj  h,j�  j}  j�  jd  j�  j-  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK K �r�	  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK K �r�	  h	hj  h,j�  j�  j�  j8  j�  j�  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK K �r�	  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK K �r�	  h	hj  h,j�  j�  j�  j$  j�  j�  j"  Nubh)�r�	  }r�	  (hKhK K�r�	  hK K �r�	  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hK K�r�	  h	hj  h,j�  h�j�  h�j�  j  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hK
K�r�	  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hKK�r�	  h	hj  h,j�  h�j�  h�j�  j  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hKK�r�	  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hKK�r�	  h	hj  h,j�  h�j�  j  j�  j  j"  Nubh)�r�	  }r�	  (hK hKK�r�	  hKK�r 
  h	hj  h,j�  h�j�  j.  j�  j  j"  Nubh)�r
  }r
  (hK hKK�r
  hKK	�r
  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hK K�r
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�r	
  }r

  (hK hK
K�r
  hKK�r
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hKK�r
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hKK�r
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hKK�r
  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hKK�r
  h	hj  h,j�  j�  j�  j.  j�  j  j"  Nubh)�r
  }r
  (hK hK
K�r
  hKK	�r 
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r!
  }r"
  (hK hKK�r#
  hK K�r$
  h	hj  h,j�  h�j�  h�j�  j  j"  Nubh)�r%
  }r&
  (hK hKK�r'
  hKK�r(
  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r)
  }r*
  (hK hKK�r+
  hKK�r,
  h	hj  h,j�  h�j�  j  j�  j  j"  Nubh)�r-
  }r.
  (hK hKK�r/
  hKK�r0
  h	hj  h,j�  h�j�  j.  j�  j  j"  Nubh)�r1
  }r2
  (hK hKK�r3
  hKK	�r4
  h	hj  h,j�  h�j�  j�  j�  j  j"  Nubh)�r5
  }r6
  (hK hKK�r7
  hK K�r8
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�r9
  }r:
  (hK hKK�r;
  hKK�r<
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�r=
  }r>
  (hK hKK�r?
  hK
K�r@
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�rA
  }rB
  (hK hKK�rC
  hKK�rD
  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�rE
  }rF
  (hK hKK�rG
  hKK�rH
  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�rI
  }rJ
  (hK hKK�rK
  hKK�rL
  h	hj  h,j�  j�  j�  j.  j�  j  j"  Nubh)�rM
  }rN
  (hK hKK�rO
  hKK	�rP
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�rQ
  }rR
  (hK hKK�rS
  hK K�rT
  h	hj  h,j�  j  j�  h�j�  j  j"  Nubh)�rU
  }rV
  (hK hKK�rW
  hK
K�rX
  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�rY
  }rZ
  (hK hKK�r[
  hKK�r\
  h	hj  h,j�  j  j�  h�j�  j  j"  Nubh)�r]
  }r^
  (hK hKK�r_
  hKK�r`
  h	hj  h,j�  j  j�  j.  j�  j  j"  Nubh)�ra
  }rb
  (hK hKK�rc
  hKK	�rd
  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�re
  }rf
  (hKhKK	�rg
  hKK�rh
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�ri
  }rj
  (hKhKK	�rk
  hK	K
�rl
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�rm
  }rn
  (hKhKK	�ro
  hKK�rp
  h	hj  h,j�  j�  j�  j�  j�  jr  j"  Nubh)�rq
  }rr
  (hKhKK	�rs
  hKK�rt
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�ru
  }rv
  (hKhK	K
�rw
  hKK�rx
  h	hj  h,j�  j  j�  j  j�  jr  j"  Nubh)�ry
  }rz
  (hKhK	K
�r{
  hKK�r|
  h	hj  h,j�  j  j�  j  j�  jr  j"  Nubh)�r}
  }r~
  (hKhKK�r
  hKK�r�
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK	�r�
  h	hj  h,j�  j�  j�  j�  j�  jr  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK	K
�r�
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j  j�  jr  j"  Nubh)�r�
  }r�
  (hKhK	K�r�
  hKK�r�
  h	hj  h,j�  j+  j�  j$  j�  j�  j"  Nubh)�r�
  }r�
  (hKhK	K�r�
  hKK�r�
  h	hj  h,j�  j+  j�  j  j�  j�  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj  h,j�  j�  j�  j5  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j;  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK	�r�
  h	hj  h,j�  j�  j�  jF  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  jP  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  jY  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK
K�r�
  h	hj  h,j�  j�  j�  jJ  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  jT  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j�  j�  jZ  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK �r�
  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj  h,j�  j;  j�  j5  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK	�r�
  h	hj  h,j�  j;  j�  jF  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  jP  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  jY  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  j�  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK
K�r�
  h	hj  h,j�  j;  j�  jJ  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  jT  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj  h,j�  j;  j�  jZ  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK �r�
  h	hj  h,j�  j;  j�  j  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj  h,j�  j�  j�  j5  j�  j  j"  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r   h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j;  j�  j  j"  Nubh)�r	  }r
  (hKhKK�r  hKK	�r  h	hj  h,j�  j�  j�  jF  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  jP  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  jY  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hK
K�r   h	hj  h,j�  j�  j�  jJ  j�  j  j"  Nubh)�r!  }r"  (hKhKK�r#  hKK�r$  h	hj  h,j�  j�  j�  jT  j�  j  j"  Nubh)�r%  }r&  (hKhKK�r'  hKK�r(  h	hj  h,j�  j�  j�  jZ  j�  j  j"  Nubh)�r)  }r*  (hKhKK�r+  hKK �r,  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r-  }r.  (hKhKK	�r/  hK K�r0  h	hj  h,j�  jF  j�  j5  j�  j  j"  Nubh)�r1  }r2  (hKhKK	�r3  hKK�r4  h	hj  h,j�  jF  j�  j�  j�  j  j"  Nubh)�r5  }r6  (hKhKK	�r7  hKK�r8  h	hj  h,j�  jF  j�  j�  j�  j  j"  Nubh)�r9  }r:  (hKhKK	�r;  hKK�r<  h	hj  h,j�  jF  j�  j;  j�  j  j"  Nubh)�r=  }r>  (hKhKK	�r?  hKK�r@  h	hj  h,j�  jF  j�  j�  j�  j  j"  Nubh)�rA  }rB  (hKhKK	�rC  hKK�rD  h	hj  h,j�  jF  j�  jP  j�  j  j"  Nubh)�rE  }rF  (hKhKK	�rG  hKK�rH  h	hj  h,j�  jF  j�  jY  j�  j  j"  Nubh)�rI  }rJ  (hKhKK	�rK  hKK�rL  h	hj  h,j�  jF  j�  j�  j�  j  j"  Nubh)�rM  }rN  (hKhKK	�rO  hK
K�rP  h	hj  h,j�  jF  j�  jJ  j�  j  j"  Nubh)�rQ  }rR  (hKhKK	�rS  hKK�rT  h	hj  h,j�  jF  j�  jT  j�  j  j"  Nubh)�rU  }rV  (hKhKK	�rW  hKK�rX  h	hj  h,j�  jF  j�  jZ  j�  j  j"  Nubh)�rY  }rZ  (hKhKK	�r[  hKK �r\  h	hj  h,j�  jF  j�  j  j�  j  j"  Nubh)�r]  }r^  (hKhKK�r_  hK K�r`  h	hj  h,j�  j�  j�  j5  j�  j  j"  Nubh)�ra  }rb  (hKhKK�rc  hKK�rd  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�re  }rf  (hKhKK�rg  hKK�rh  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�ri  }rj  (hKhKK�rk  hKK�rl  h	hj  h,j�  j�  j�  j;  j�  j  j"  Nubh)�rm  }rn  (hKhKK�ro  hKK�rp  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�rq  }rr  (hKhKK�rs  hKK	�rt  h	hj  h,j�  j�  j�  jF  j�  j  j"  Nubh)�ru  }rv  (hKhKK�rw  hKK�rx  h	hj  h,j�  j�  j�  jP  j�  j  j"  Nubh)�ry  }rz  (hKhKK�r{  hKK�r|  h	hj  h,j�  j�  j�  jY  j�  j  j"  Nubh)�r}  }r~  (hKhKK�r  hKK�r�  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK
K�r�  h	hj  h,j�  j�  j�  jJ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  j�  j�  jT  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  j�  j�  jZ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK �r�  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  jP  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj  h,j�  jP  j�  jF  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  jY  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK
K�r�  h	hj  h,j�  jP  j�  jJ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  jT  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jP  j�  jZ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK �r�  h	hj  h,j�  jP  j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  jY  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj  h,j�  jY  j�  jF  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  jP  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK
K�r�  h	hj  h,j�  jY  j�  jJ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  jT  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jY  j�  jZ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK �r�  h	hj  h,j�  jY  j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  j�  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r   h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j;  j�  j  j"  Nubh)�r	  }r
  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK	�r  h	hj  h,j�  j�  j�  jF  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  jP  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  j�  j�  jY  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hK
K�r   h	hj  h,j�  j�  j�  jJ  j�  j  j"  Nubh)�r!  }r"  (hKhKK�r#  hKK�r$  h	hj  h,j�  j�  j�  jT  j�  j  j"  Nubh)�r%  }r&  (hKhKK�r'  hKK�r(  h	hj  h,j�  j�  j�  jZ  j�  j  j"  Nubh)�r)  }r*  (hKhKK�r+  hKK �r,  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�r-  }r.  (hKhKK�r/  hK K�r0  h	hj  h,j�  ji  j�  jc  j�  j�  j"  Nubh)�r1  }r2  (hKhKK�r3  hKK�r4  h	hj  h,j�  ji  j�  js  j�  j�  j"  Nubh)�r5  }r6  (hKhKK�r7  hK K�r8  h	hj  h,j�  js  j�  jc  j�  j�  j"  Nubh)�r9  }r:  (hKhKK�r;  hKK�r<  h	hj  h,j�  js  j�  ji  j�  j�  j"  Nubh)�r=  }r>  (hKhKK�r?  hK K�r@  h	hj  h,j�  j�  j�  j}  j�  j-  j"  Nubh)�rA  }rB  (hKhKK�rC  hKK�rD  h	hj  h,j�  j�  j�  j�  j�  j-  j"  Nubh)�rE  }rF  (hKhKK�rG  hKK�rH  h	hj  h,j�  j�  j�  j�  j�  j-  j"  Nubh)�rI  }rJ  (hKhKK�rK  hK K�rL  h	hj  h,j�  j�  j�  j}  j�  j-  j"  Nubh)�rM  }rN  (hKhKK�rO  hKK�rP  h	hj  h,j�  j�  j�  j�  j�  j-  j"  Nubh)�rQ  }rR  (hKhKK�rS  hKK�rT  h	hj  h,j�  j�  j�  j�  j�  j-  j"  Nubh)�rU  }rV  (hKhKK�rW  hK	K�rX  h	hj  h,j�  j�  j�  j	  j�  j-  j"  Nubh)�rY  }rZ  (hKhKK�r[  hKK�r\  h	hj  h,j�  j�  j�  jd  j�  j-  j"  Nubh)�r]  }r^  (hKhKK�r_  hK K�r`  h	hj  h,j�  j�  j�  j}  j�  j-  j"  Nubh)�ra  }rb  (hKhKK�rc  hKK�rd  h	hj  h,j�  j�  j�  j�  j�  j-  j"  Nubh)�re  }rf  (hKhKK�rg  hK	K�rh  h	hj  h,j�  j�  j�  j	  j�  j-  j"  Nubh)�ri  }rj  (hKhKK�rk  hKK�rl  h	hj  h,j�  j�  j�  jd  j�  j-  j"  Nubh)�rm  }rn  (hKhK	K�ro  hK K�rp  h	hj  h,j�  j	  j�  j}  j�  j-  j"  Nubh)�rq  }rr  (hKhK	K�rs  hKK�rt  h	hj  h,j�  j	  j�  j�  j�  j-  j"  Nubh)�ru  }rv  (hKhK	K�rw  hKK�rx  h	hj  h,j�  j	  j�  j�  j�  j-  j"  Nubh)�ry  }rz  (hKhK	K�r{  hKK�r|  h	hj  h,j�  j	  j�  j�  j�  j-  j"  Nubh)�r}  }r~  (hKhK	K�r  hKK�r�  h	hj  h,j�  j	  j�  jd  j�  j-  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j8  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j$  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j8  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j$  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j  j�  j8  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K�r�  h	hj  h,j�  j8  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j8  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j8  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j8  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j8  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j8  j�  j$  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j8  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hKK�r�  h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j  j�  j  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj  h,j�  j  j�  j$  j�  j�  j"  Nubh)�r�  }r�  (hKhK K �r�  hK K �r   h	hj  h,j�  j  j�  j�  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K�r  h	hj  h,j�  j$  j�  j�  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hKK�r  h	hj  h,j�  j$  j�  j�  j�  j�  j"  Nubh)�r	  }r
  (hKhK K �r  hKK�r  h	hj  h,j�  j$  j�  j�  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K �r  h	hj  h,j�  j$  j�  j  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K �r  h	hj  h,j�  j$  j�  j8  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K �r  h	hj  h,j�  j$  j�  j  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K �r  h	hj  h,j�  j$  j�  j�  j�  j�  j"  Nubh)�r  }r  (hKhK K �r  hK K�r   h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r!  }r"  (hKhK K �r#  hKK�r$  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r%  }r&  (hKhK K �r'  hKK�r(  h	hj  h,j�  j�  j�  j�  j�  j�  j"  Nubh)�r)  }r*  (hKhK K �r+  hK K �r,  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r-  }r.  (hKhK K �r/  hK K �r0  h	hj  h,j�  j�  j�  j8  j�  j�  j"  Nubh)�r1  }r2  (hKhK K �r3  hK K �r4  h	hj  h,j�  j�  j�  j  j�  j�  j"  Nubh)�r5  }r6  (hKhK K �r7  hK K �r8  h	hj  h,j�  j�  j�  j$  j�  j�  j"  Nubh)�r9  }r:  (hK hKK�r;  hK K�r<  h	hj  h,j�  j.  j�  h�j�  j  j"  Nubh)�r=  }r>  (hK hKK�r?  hK
K�r@  h	hj  h,j�  j.  j�  j�  j�  j  j"  Nubh)�rA  }rB  (hK hKK�rC  hKK�rD  h	hj  h,j�  j.  j�  h�j�  j  j"  Nubh)�rE  }rF  (hK hKK�rG  hKK�rH  h	hj  h,j�  j.  j�  j�  j�  j  j"  Nubh)�rI  }rJ  (hK hKK�rK  hKK�rL  h	hj  h,j�  j.  j�  j  j�  j  j"  Nubh)�rM  }rN  (hK hKK�rO  hKK	�rP  h	hj  h,j�  j.  j�  j�  j�  j  j"  Nubh)�rQ  }rR  (hK hKK	�rS  hK K�rT  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�rU  }rV  (hK hKK	�rW  hKK�rX  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�rY  }rZ  (hK hKK	�r[  hK
K�r\  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�r]  }r^  (hK hKK	�r_  hKK�r`  h	hj  h,j�  j�  j�  h�j�  j  j"  Nubh)�ra  }rb  (hK hKK	�rc  hKK�rd  h	hj  h,j�  j�  j�  j�  j�  j  j"  Nubh)�re  }rf  (hK hKK	�rg  hKK�rh  h	hj  h,j�  j�  j�  j  j�  j  j"  Nubh)�ri  }rj  (hK hKK	�rk  hKK�rl  h	hj  h,j�  j�  j�  j.  j�  j  j"  Nubh)�rm  }rn  (hKhKK�ro  hKK�rp  h	hj  h,j�  j  j�  j  j�  jr  j"  Nubh)�rq  }rr  (hKhKK�rs  hKK	�rt  h	hj  h,j�  j  j�  j�  j�  jr  j"  Nubh)�ru  }rv  (hKhKK�rw  hK	K
�rx  h	hj  h,j�  j  j�  j  j�  jr  j"  Nubh)�ry  }rz  (hKhKK�r{  hKK�r|  h	hj  h,j�  j  j�  j�  j�  jr  j"  Nubh)�r}  }r~  (hKhK
K�r  hK K�r�  h	hj  h,j�  jJ  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  jP  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  jY  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK�r�  h	hj  h,j�  jJ  j�  jT  j�  j  j"  Nubh)�r�  }r�  (hKhK
K�r�  hKK �r�  h	hj  h,j�  jJ  j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  jT  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  jP  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  jY  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jT  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hK
K�r�  h	hj  h,j�  jT  j�  jJ  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK �r�  h	hj  h,j�  jT  j�  j  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  j  j�  j$  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK	K�r�  h	hj  h,j�  j  j�  j+  j�  j�  j"  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj  h,j�  jZ  j�  j5  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  j;  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj  h,j�  jZ  j�  jF  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  j�  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj  h,j�  jZ  j�  jP  j�  j  j"  Nubh)�r�  }r�  (hKhKK�r�  hKK�r   h	hj  h,j�  jZ  j�  jY  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj  h,j�  jZ  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hK
K�r  h	hj  h,j�  jZ  j�  jJ  j�  j  j"  Nubh)�r	  }r
  (hKhKK�r  hKK�r  h	hj  h,j�  jZ  j�  jT  j�  j  j"  Nubh)�r  }r  (hKhKK�r  hKK �r  h	hj  h,j�  jZ  j�  j  j�  j  j"  Nubh)�r  }r  (hKhKK �r  hK K�r  h	hj  h,j�  j  j�  j5  j�  j  j"  Nubh)�r  }r  (hKhKK �r  hKK�r  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK �r  hKK�r  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�r  }r  (hKhKK �r  hKK�r   h	hj  h,j�  j  j�  j;  j�  j  j"  Nubh)�r!  }r"  (hKhKK �r#  hKK�r$  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�r%  }r&  (hKhKK �r'  hKK	�r(  h	hj  h,j�  j  j�  jF  j�  j  j"  Nubh)�r)  }r*  (hKhKK �r+  hKK�r,  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�r-  }r.  (hKhKK �r/  hKK�r0  h	hj  h,j�  j  j�  jP  j�  j  j"  Nubh)�r1  }r2  (hKhKK �r3  hKK�r4  h	hj  h,j�  j  j�  jY  j�  j  j"  Nubh)�r5  }r6  (hKhKK �r7  hKK�r8  h	hj  h,j�  j  j�  j�  j�  j  j"  Nubh)�r9  }r:  (hKhKK �r;  hK
K�r<  h	hj  h,j�  j  j�  jJ  j�  j  j"  Nubh)�r=  }r>  (hKhKK �r?  hKK�r@  h	hj  h,j�  j  j�  jT  j�  j  j"  Nubh)�rA  }rB  (hKhKK �rC  hKK�rD  h	hj  h,j�  j  j�  jZ  j�  j  j"  Nubh)�rE  }rF  (hKhKK�rG  hK K�rH  h	hj  h,j�  jd  j�  j}  j�  j-  j"  Nubh)�rI  }rJ  (hKhKK�rK  hKK�rL  h	hj  h,j�  jd  j�  j�  j�  j-  j"  Nubh)�rM  }rN  (hKhKK�rO  hKK�rP  h	hj  h,j�  jd  j�  j�  j�  j-  j"  Nubh)�rQ  }rR  (hKhKK�rS  hKK�rT  h	hj  h,j�  jd  j�  j�  j�  j-  j"  Nubh)�rU  }rV  (hKhKK�rW  hK	K�rX  h	hj  h,j�  jd  j�  j	  j�  j-  j"  Nube.