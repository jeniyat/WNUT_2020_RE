�]q (c__main__
Relation
q)�q}q(X   sent_idxqK X   arg1qK K�qX   arg2qKK�qX   pq	c__main__
ProtoFile
q
)�q}q(X   filenameqXN   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/train_full/protocol_533qX   basenameqX   protocol_533qX   protocol_nameqhX	   text_fileqXR   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/train_full/protocol_533.txtqX   ann_fileqXR   /home/jeniya/WLP-RE-LR-baseline/WLP-Parser/WLP-Dataset/train_full/protocol_533.annqX	   tokenizerqcsacremoses.tokenize
MosesTokenizer
q)�q}q(X   langqX   enqX   NONBREAKING_PREFIXESq]q(X   AqX   BqX   Cq X   Dq!X   Eq"X   Fq#X   Gq$X   Hq%X   Iq&X   Jq'X   Kq(X   Lq)X   Mq*X   Nq+X   Oq,X   Pq-X   Qq.X   Rq/X   Sq0X   Tq1X   Uq2X   Vq3X   Wq4X   Xq5X   Yq6X   Zq7X   Adjq8X   Admq9X   Advq:X   Asstq;X   Bartq<X   Bldgq=X   Brigq>X   Brosq?X   Captq@X   CmdrqAX   ColqBX   ComdrqCX   ConqDX   CorpqEX   CplqFX   DRqGX   DrqHX   DrsqIX   EnsqJX   GenqKX   GovqLX   HonqMX   HrqNX   HospqOX   InspqPX   LtqQX   MMqRX   MRqSX   MRSqTX   MSqUX   MajqVX   MessrsqWX   MlleqXX   MmeqYX   MrqZX   Mrsq[X   Msq\X   Msgrq]X   Opq^X   Ordq_X   Pfcq`X   PhqaX   ProfqbX   PvtqcX   RepqdX   RepsqeX   ResqfX   RevqgX   RtqhX   SenqiX   SensqjX   SfcqkX   SgtqlX   SrqmX   StqnX   SuptqoX   SurgqpX   vqqX   vsqrX   i.eqsX   revqtX   e.gquX   No #NUMERIC_ONLY#qvX   NosqwX   Art #NUMERIC_ONLY#qxX   NrqyX   pp #NUMERIC_ONLY#qzX   Janq{X   Febq|X   Marq}X   Aprq~X   JunqX   Julq�X   Augq�X   Sepq�X   Octq�X   Novq�X   Decq�eX   NUMERIC_ONLY_PREFIXESq�]q�(X   Noq�X   Artq�X   ppq�eubX   linesq�]q�(X   From siRNA to shRNA
q�X7   Anchor the siRNA sequence (19nt) on your target gene.
q�X_   Extract 2nt of upstream +19nt siRNA + 1nt of downstream, which is totally 22nt of nucleotides.
q�XP   Antisense shRNA: Insert the reverse complement of the 22nt into shRNA backbone.
q�X@   Sense shRNA: Insert the 22nt (from step 2) into shRNA backbone.
q�Xm   Replace the 1st nt with other nucleotide (replace A with G and vice versa; replace C with T and vice versa).
q�X&   Now your shRNA designing job is done.
q�X   You see, how easy it is!
q�XD   Synthesis the shRNA nucleotides and insert them into pTRIPZ vector.
q�XI   You may also translocate the shRNA from pTRIPZ vector into pGIPZ vector.
q�eX   textq�Xs  From siRNA to shRNA
Anchor the siRNA sequence (19nt) on your target gene.
Extract 2nt of upstream +19nt siRNA + 1nt of downstream, which is totally 22nt of nucleotides.
Antisense shRNA: Insert the reverse complement of the 22nt into shRNA backbone.
Sense shRNA: Insert the 22nt (from step 2) into shRNA backbone.
Replace the 1st nt with other nucleotide (replace A with G and vice versa; replace C with T and vice versa).
Now your shRNA designing job is done.
You see, how easy it is!
Synthesis the shRNA nucleotides and insert them into pTRIPZ vector.
You may also translocate the shRNA from pTRIPZ vector into pGIPZ vector.
q�X   annq�]q�(X   T1	Action 20 26	Anchor
q�X   E1 Action:T1 Acts-on:T8 Site:T9q�X   T2	Action 74 81	Extract
q�X2   E2 Action:T2 Acts-on:T13 Acts-on2:T15 Acts-on3:T17q�X   T3	Action 186 192	Insert
q�X!   E3 Action:T3 Acts-on:T11 Site:T12q�X   T4	Action 262 268	Insert
q�X!   E4 Action:T4 Acts-on:T19 Site:T20q�X   T5	Action 313 320	Replace
q�X"   E5 Action:T5 Acts-on:T22 Using:T21q�X   T6	Action 485 494	Synthesis
q�X   E6 Action:T6 Acts-on:T23q�X   T7	Action 521 527	insert
q�X!   E7 Action:T7 Acts-on:T24 Site:T25q�X    T8	Reagent 31 45	siRNA sequence
q�X   T9	Reagent 61 72	target gene
q�X   T10	Amount 82 85	2nt
q�X'   T11	Reagent 197 215	reverse complement
q�X#   T12	Reagent 233 247	shRNA backbone
q�X   T13	Reagent 89 97	upstream
q�X   R1 Measure Arg1:T13 Arg2:T10q�X   T14	Amount 112 115	1nt
q�X   T15	Reagent 119 129	downstream
q�X   R2 Measure Arg1:T15 Arg2:T14q�X   T16	Amount 99 103	19nt
q�X   T17	Reagent 104 109	siRNA
q�X   R3 Measure Arg1:T17 Arg2:T16q�X   T18	Amount 273 277	22nt
q�X    T19	Reagent 279 290	from step 2
q�X   R4 Measure Arg1:T19 Arg2:T18q�X#   T20	Reagent 297 311	shRNA backbone
q�X%   T21	Reagent 337 353	other nucleotide
q�X   T22	Reagent 325 331	1st nt
q�X&   T23	Reagent 499 516	shRNA nucleotides
q�X   T24	Mention 528 532	them
q�X%   R5 Coreference-Link Arg1:T24 Arg2:T23q�X"   T25	Reagent 538 551	pTRIPZ vector
q�X   T26	Reagent 5 10	siRNA
q�X   T27	Reagent 14 19	shRNA
q�X   T28	Generic-Measure 47 51	19nt
q�X$   R6 Coreference-Link Arg1:T8 Arg2:T28q�X!   T29	Generic-Measure 148 152	22nt
q�X    T30	Reagent 156 167	nucleotides
q�X   R7 Measure Arg1:T30 Arg2:T29q�X   T31	Modifier 140 147	totally
q�X   R8 Mod-Link Arg1:T29 Arg2:T31q�X$   T32	Reagent 169 184	Antisense shRNA
q�X   T33	Reagent 223 227	22nt
q�X   R9 Meronym Arg1:T11 Arg2:T33q�X    T34	Reagent 249 260	Sense shRNA
q�X   T35	Action 355 362	replace
q�X#   E8 Action:T35 Acts-on:T36 Using:T37q�X   T36	Reagent 363 364	A
q�X   T37	Reagent 370 371	G
q�X$   T38	Modifier 372 386	and vice versa
q�X   R10 Mod-Link Arg1:T35 Arg2:T38q�X   T39	Action 388 395	replace
q�X&   E9 Action:T39 Acts-on:T40 Acts-on2:T41q�X   T40	Reagent 396 397	C
q�X   T41	Reagent 403 404	T
q�X$   T42	Modifier 405 419	and vice versa
q�X   R11 Mod-Link Arg1:T39 Arg2:T42q�X   T43	Reagent 431 436	shRNA
q�X!   T44	Method 437 450	designing job
q�X   T45	Modifier 454 458	done
q�X   T46	Mention 478 480	it
q�X   T47	Modifier 473 477	easy
q�X   R12 Mod-Link Arg1:T46 Arg2:T47q�X   T48	Action 566 577	translocate
q�X-   E10 Action:T48 Acts-on:T49 Site:T50 Site2:T51q�X   T49	Reagent 582 587	shRNA
q�X"   T50	Reagent 593 606	pTRIPZ vector
q�X!   T51	Reagent 612 624	pGIPZ vector
q�eX   statusq�X   linksq�]q�(c__main__
Link
q�(X   E1q�X   Acts-onq�c__main__
Tag
q�(X   T1q�X   Actionq�KK]q�X   Anchorq�atq�q�h�(X   T8q�X   Reagentq�KK-]q�(X   siRNAq�X   sequenceq�etq��q�tq��q�h�(h�X   Siteq�h�h�(X   T9q�X   Reagentq�K=KH]q�(X   targetq�X   geneq�etr   �r  tr  �r  h�(X   E2r  X   Acts-onr  h�(X   T2r  X   Actionr  KJKQ]r  X   Extractr	  atr
  �r  h�(X   T13r  X   Reagentr  KYKa]r  X   upstreamr  atr  �r  tr  �r  h�(j  X   Acts-on2r  j  h�(X   T15r  X   Reagentr  KwK�]r  X
   downstreamr  atr  �r  tr  �r  h�(j  X   Acts-on3r  j  h�(X   T17r  X   Reagentr  KhKm]r   X   siRNAr!  atr"  �r#  tr$  �r%  h�(X   E3r&  X   Acts-onr'  h�(X   T3r(  X   Actionr)  K�K�]r*  X   Insertr+  atr,  �r-  h�(X   T11r.  X   Reagentr/  K�K�]r0  (X   reverser1  X
   complementr2  etr3  �r4  tr5  �r6  h�(j&  X   Siter7  j-  h�(X   T12r8  X   Reagentr9  K�K�]r:  (X   shRNAr;  X   backboner<  etr=  �r>  tr?  �r@  h�(X   E4rA  X   Acts-onrB  h�(X   T4rC  X   ActionrD  MM]rE  X   InsertrF  atrG  �rH  h�(X   T19rI  X   ReagentrJ  MM"]rK  (X   fromrL  X   steprM  X   2rN  etrO  �rP  trQ  �rR  h�(jA  X   SiterS  jH  h�(X   T20rT  X   ReagentrU  M)M7]rV  (X   shRNArW  X   backbonerX  etrY  �rZ  tr[  �r\  h�(X   E5r]  X   Acts-onr^  h�(X   T5r_  X   Actionr`  M9M@]ra  X   Replacerb  atrc  �rd  h�(X   T22re  X   Reagentrf  MEMK]rg  (X   1strh  X   ntri  etrj  �rk  trl  �rm  h�(j]  X   Usingrn  jd  h�(X   T21ro  X   Reagentrp  MQMa]rq  (X   otherrr  X
   nucleotiders  etrt  �ru  trv  �rw  h�(X   E6rx  X   Acts-onry  h�(X   T6rz  X   Actionr{  M�M�]r|  X	   Synthesisr}  atr~  �r  h�(X   T23r�  X   Reagentr�  M�M]r�  (X   shRNAr�  X   nucleotidesr�  etr�  �r�  tr�  �r�  h�(X   E7r�  X   Acts-onr�  h�(X   T7r�  X   Actionr�  M	M]r�  X   insertr�  atr�  �r�  h�(X   T24r�  X   Mentionr�  MM]r�  X   themr�  atr�  �r�  tr�  �r�  h�(j�  X   Siter�  j�  h�(X   T25r�  X   Reagentr�  MM']r�  (X   pTRIPZr�  X   vectorr�  etr�  �r�  tr�  �r�  h�(X   R1r�  X   Measurer�  j  h�(X   T10r�  X   Amountr�  KRKU]r�  X   2ntr�  atr�  �r�  tr�  �r�  h�(X   R2r�  X   Measurer�  j  h�(X   T14r�  X   Amountr�  KpKs]r�  X   1ntr�  atr�  �r�  tr�  �r�  h�(X   R3r�  X   Measurer�  j#  h�(X   T16r�  X   Amountr�  KcKg]r�  X   19ntr�  atr�  �r�  tr�  �r�  h�(X   R4r�  X   Measurer�  jP  h�(X   T18r�  X   Amountr�  MM]r�  X   22ntr�  atr�  �r�  tr�  �r�  h�(X   R5r�  X   Coreference-Linkr�  j�  j�  tr�  �r�  h�(X   R6r�  X   Coreference-Linkr�  h�h�(X   T28r�  X   Generic-Measurer�  K/K3]r�  X   19ntr�  atr�  �r�  tr�  �r�  h�(X   R7r�  X   Measurer�  h�(X   T30r�  X   Reagentr�  K�K�]r�  X   nucleotidesr�  atr�  �r�  h�(X   T29r�  X   Generic-Measurer�  K�K�]r�  X   22ntr�  atr�  �r�  tr�  �r�  h�(X   R8r�  X   Mod-Linkr�  j�  h�(X   T31r�  X   Modifierr�  K�K�]r�  X   totallyr�  atr�  �r�  tr�  �r�  h�(X   R9r�  X   Meronymr�  j4  h�(X   T33r�  X   Reagentr�  K�K�]r�  X   22ntr�  atr�  �r�  tr�  �r�  h�(X   E8r�  X   Acts-onr�  h�(X   T35r�  X   Actionr   McMj]r  X   replacer  atr  �r  h�(X   T36r  X   Reagentr  MkMl]r  hatr  �r	  tr
  �r  h�(j�  X   Usingr  j  h�(X   T37r  X   Reagentr  MrMs]r  h$atr  �r  tr  �r  h�(X   R10r  X   Mod-Linkr  j  h�(X   T38r  X   Modifierr  MtM�]r  (X   andr  X   vicer  X   versar  etr  �r  tr  �r  h�(X   E9r   X   Acts-onr!  h�(X   T39r"  X   Actionr#  M�M�]r$  X   replacer%  atr&  �r'  h�(X   T40r(  X   Reagentr)  M�M�]r*  h atr+  �r,  tr-  �r.  h�(j   X   Acts-on2r/  j'  h�(X   T41r0  X   Reagentr1  M�M�]r2  h1atr3  �r4  tr5  �r6  h�(X   R11r7  X   Mod-Linkr8  j'  h�(X   T42r9  X   Modifierr:  M�M�]r;  (X   andr<  X   vicer=  X   versar>  etr?  �r@  trA  �rB  h�(X   R12rC  X   Mod-LinkrD  h�(X   T46rE  X   MentionrF  M�M�]rG  X   itrH  atrI  �rJ  h�(X   T47rK  X   ModifierrL  M�M�]rM  X   easyrN  atrO  �rP  trQ  �rR  h�(X   E10rS  X   Acts-onrT  h�(X   T48rU  X   ActionrV  M6MA]rW  X   translocaterX  atrY  �rZ  h�(X   T49r[  X   Reagentr\  MFMK]r]  X   shRNAr^  atr_  �r`  tra  �rb  h�(jS  X   Siterc  jZ  h�(X   T50rd  X   Reagentre  MQM^]rf  (X   pTRIPZrg  X   vectorrh  etri  �rj  trk  �rl  h�(jS  X   Site2rm  jZ  h�(X   T51rn  X   Reagentro  MdMp]rp  (X   pGIPZrq  X   vectorrr  etrs  �rt  tru  �rv  eX   headingrw  ]rx  (X   Fromry  X   siRNArz  X   tor{  X   shRNAr|  eX   sentsr}  ]r~  (]r  (X   Anchorr�  X   ther�  X   siRNAr�  X   sequencer�  X   (r�  X   19ntr�  X   )r�  X   onr�  X   yourr�  X   targetr�  X   gener�  X   .r�  e]r�  (X   Extractr�  X   2ntr�  X   ofr�  X   upstreamr�  X   +r�  X   19ntr�  X   siRNAr�  j�  X   1ntr�  X   ofr�  X
   downstreamr�  X   ,r�  X   whichr�  X   isr�  X   totallyr�  X   22ntr�  X   ofr�  X   nucleotidesr�  j�  e]r�  (X	   Antisenser�  X   shRNAr�  X   :r�  X   Insertr�  X   ther�  X   reverser�  X
   complementr�  X   ofr�  X   ther�  X   22ntr�  X   intor�  X   shRNAr�  X   backboner�  j�  e]r�  (X   Senser�  X   shRNAr�  j�  X   Insertr�  X   ther�  X   22ntr�  j�  X   fromr�  X   stepr�  jN  j�  X   intor�  X   shRNAr�  X   backboner�  j�  e]r�  (X   Replacer�  X   ther�  X   1str�  X   ntr�  X   withr�  X   otherr�  X
   nucleotider�  j�  X   replacer�  hX   withr�  h$X   andr�  X   vicer�  X   versar�  X   ;r�  X   replacer�  h X   withr�  h1X   andr�  X   vicer�  X   versar�  j�  j�  e]r�  (X   Nowr�  X   yourr�  X   shRNAr�  X	   designingr�  X   jobr�  X   isr�  X   doner�  j�  e]r�  (X   Your�  X   seer�  j�  X   howr�  X   easyr�  X   itr�  X   isr�  X   !r�  e]r�  (X	   Synthesisr�  X   ther�  X   shRNAr�  X   nucleotidesr�  X   andr�  X   insertr�  X   themr�  X   intor�  X   pTRIPZr�  X   vectorr�  j�  e]r�  (X   Your�  X   mayr�  X   alsor�  X   translocater�  X   ther�  X   shRNAr�  X   fromr�  X   pTRIPZr�  X   vectorr�  X   intor�  X   pGIPZr�  X   vectorr�  j�  eeX   tagsr�  ]r�  (h�j  j-  jH  jd  j  j�  h�j  j�  j4  j>  j  j�  j  j�  j#  j�  jP  jZ  ju  jk  j�  j�  j�  h�(X   T26r�  X   Reagentr�  KK
]r�  X   siRNAr�  atr�  �r�  h�(X   T27r�  X   Reagentr�  KK]r�  X   shRNAr�  atr�  �r�  j�  j�  j�  j�  h�(X   T32r   X   Reagentr  K�K�]r  (X	   Antisenser  X   shRNAr  etr  �r  j�  h�(X   T34r  X   Reagentr  K�M]r	  (X   Senser
  X   shRNAr  etr  �r  j  j	  j  j  j'  j,  j4  j@  h�(X   T43r  X   Reagentr  M�M�]r  X   shRNAr  atr  �r  h�(X   T44r  X   Methodr  M�M�]r  (X	   designingr  X   jobr  etr  �r  h�(X   T45r  X   Modifierr  M�M�]r  X   doner  atr  �r   jJ  jP  jZ  j`  jj  jt  eX   unique_tagsr!  cbuiltins
set
r"  ]r#  (j�  h�j  j�  j�  h�j�  e�r$  Rr%  X   tag_0_idr&  X   T0r'  X
   tag_0_namer(  h,X   tokens2dr)  ]r*  (]r+  (c__main__
Token
r,  )�r-  }r.  (X   wordr/  h�X   labelr0  X   B-Actionr1  X   originalr2  h�X   feature_valuesr3  Nubj,  )�r4  }r5  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r6  }r7  (j/  h�j0  X	   B-Reagentr8  j2  h�j3  Nubj,  )�r9  }r:  (j/  h�j0  X	   I-Reagentr;  j2  h�j3  Nubj,  )�r<  }r=  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r>  }r?  (j/  j�  j0  X   B-Generic-Measurer@  j2  j�  j3  Nubj,  )�rA  }rB  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rC  }rD  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rE  }rF  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rG  }rH  (j/  h�j0  X	   B-ReagentrI  j2  h�j3  Nubj,  )�rJ  }rK  (j/  h�j0  X	   I-ReagentrL  j2  h�j3  Nubj,  )�rM  }rN  (j/  j�  j0  h,j2  j�  j3  Nube]rO  (j,  )�rP  }rQ  (j/  j	  j0  X   B-ActionrR  j2  j	  j3  Nubj,  )�rS  }rT  (j/  j�  j0  X   B-AmountrU  j2  j�  j3  Nubj,  )�rV  }rW  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rX  }rY  (j/  j  j0  X	   B-ReagentrZ  j2  j  j3  Nubj,  )�r[  }r\  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r]  }r^  (j/  j�  j0  X   B-Amountr_  j2  j�  j3  Nubj,  )�r`  }ra  (j/  j!  j0  X	   B-Reagentrb  j2  j!  j3  Nubj,  )�rc  }rd  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�re  }rf  (j/  j�  j0  X   B-Amountrg  j2  j�  j3  Nubj,  )�rh  }ri  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rj  }rk  (j/  j  j0  X	   B-Reagentrl  j2  j  j3  Nubj,  )�rm  }rn  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�ro  }rp  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rq  }rr  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rs  }rt  (j/  j�  j0  X
   B-Modifierru  j2  j�  j3  Nubj,  )�rv  }rw  (j/  j�  j0  X   B-Generic-Measurerx  j2  j�  j3  Nubj,  )�ry  }rz  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r{  }r|  (j/  j�  j0  X	   B-Reagentr}  j2  j�  j3  Nubj,  )�r~  }r  (j/  j�  j0  h,j2  j�  j3  Nube]r�  (j,  )�r�  }r�  (j/  j  j0  X	   B-Reagentr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j  j0  X	   I-Reagentr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j+  j0  X   B-Actionr�  j2  j+  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j1  j0  X	   B-Reagentr�  j2  j1  j3  Nubj,  )�r�  }r�  (j/  j2  j0  X	   I-Reagentr�  j2  j2  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  X	   B-Reagentr�  j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j;  j0  X	   B-Reagentr�  j2  j;  j3  Nubj,  )�r�  }r�  (j/  j<  j0  X	   I-Reagentr�  j2  j<  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nube]r�  (j,  )�r�  }r�  (j/  j
  j0  X	   B-Reagentr�  j2  j
  j3  Nubj,  )�r�  }r�  (j/  j  j0  X	   I-Reagentr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  jF  j0  X   B-Actionr�  j2  jF  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  X   B-Amountr�  j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  jL  j0  X	   B-Reagentr�  j2  jL  j3  Nubj,  )�r�  }r�  (j/  jM  j0  X	   I-Reagentr�  j2  jM  j3  Nubj,  )�r�  }r�  (j/  jN  j0  X	   I-Reagentr�  j2  jN  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  jW  j0  X	   B-Reagentr�  j2  jW  j3  Nubj,  )�r�  }r�  (j/  jX  j0  X	   I-Reagentr�  j2  jX  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nube]r�  (j,  )�r�  }r�  (j/  jb  j0  X   B-Actionr�  j2  jb  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  jh  j0  X	   B-Reagentr�  j2  jh  j3  Nubj,  )�r�  }r�  (j/  ji  j0  X	   I-Reagentr�  j2  ji  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  jr  j0  X	   B-Reagentr�  j2  jr  j3  Nubj,  )�r�  }r�  (j/  js  j0  X	   I-Reagentr�  j2  js  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j  j0  X   B-Actionr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  hj0  X	   B-Reagentr�  j2  hj3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  h$j0  X	   B-Reagentr�  j2  h$j3  Nubj,  )�r�  }r�  (j/  j  j0  X
   B-Modifierr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j  j0  X
   I-Modifierr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j  j0  X
   I-Modifierr�  j2  j  j3  Nubj,  )�r�  }r�  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r�  }r�  (j/  j%  j0  X   B-Actionr�  j2  j%  j3  Nubj,  )�r�  }r�  (j/  h j0  X	   B-Reagentr�  j2  h j3  Nubj,  )�r�  }r   (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r  }r  (j/  h1j0  X	   B-Reagentr  j2  h1j3  Nubj,  )�r  }r  (j/  j<  j0  X
   B-Modifierr  j2  j<  j3  Nubj,  )�r  }r  (j/  j=  j0  X
   I-Modifierr	  j2  j=  j3  Nubj,  )�r
  }r  (j/  j>  j0  X
   I-Modifierr  j2  j>  j3  Nubj,  )�r  }r  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r  }r  (j/  j�  j0  h,j2  j�  j3  Nube]r  (j,  )�r  }r  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r  }r  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r  }r  (j/  j  j0  X	   B-Reagentr  j2  j  j3  Nubj,  )�r  }r  (j/  j  j0  X   B-Methodr  j2  j  j3  Nubj,  )�r  }r  (j/  j  j0  X   I-Methodr  j2  j  j3  Nubj,  )�r  }r   (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r!  }r"  (j/  j  j0  X
   B-Modifierr#  j2  j  j3  Nubj,  )�r$  }r%  (j/  j�  j0  h,j2  j�  j3  Nube]r&  (j,  )�r'  }r(  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r)  }r*  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r+  }r,  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r-  }r.  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r/  }r0  (j/  jN  j0  X
   B-Modifierr1  j2  jN  j3  Nubj,  )�r2  }r3  (j/  jH  j0  X	   B-Mentionr4  j2  jH  j3  Nubj,  )�r5  }r6  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r7  }r8  (j/  j�  j0  h,j2  j�  j3  Nube]r9  (j,  )�r:  }r;  (j/  j}  j0  X   B-Actionr<  j2  j}  j3  Nubj,  )�r=  }r>  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r?  }r@  (j/  j�  j0  X	   B-ReagentrA  j2  j�  j3  Nubj,  )�rB  }rC  (j/  j�  j0  X	   I-ReagentrD  j2  j�  j3  Nubj,  )�rE  }rF  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rG  }rH  (j/  j�  j0  X   B-ActionrI  j2  j�  j3  Nubj,  )�rJ  }rK  (j/  j�  j0  X	   B-MentionrL  j2  j�  j3  Nubj,  )�rM  }rN  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rO  }rP  (j/  j�  j0  X	   B-ReagentrQ  j2  j�  j3  Nubj,  )�rR  }rS  (j/  j�  j0  X	   I-ReagentrT  j2  j�  j3  Nubj,  )�rU  }rV  (j/  j�  j0  h,j2  j�  j3  Nube]rW  (j,  )�rX  }rY  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rZ  }r[  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r\  }r]  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�r^  }r_  (j/  jX  j0  X   B-Actionr`  j2  jX  j3  Nubj,  )�ra  }rb  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rc  }rd  (j/  j^  j0  X	   B-Reagentre  j2  j^  j3  Nubj,  )�rf  }rg  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rh  }ri  (j/  jg  j0  X	   B-Reagentrj  j2  jg  j3  Nubj,  )�rk  }rl  (j/  jh  j0  X	   I-Reagentrm  j2  jh  j3  Nubj,  )�rn  }ro  (j/  j�  j0  h,j2  j�  j3  Nubj,  )�rp  }rq  (j/  jq  j0  X	   B-Reagentrr  j2  jq  j3  Nubj,  )�rs  }rt  (j/  jr  j0  X	   I-Reagentru  j2  jr  j3  Nubj,  )�rv  }rw  (j/  j�  j0  h,j2  j�  j3  NubeeX   word_cntrx  K}X   f_dfry  NX   pos_tagsrz  ]r{  (]r|  (X   Anchorr}  X   INr~  X   B-PPr  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   siRNAr�  X   NNr�  X   I-NPr�  �r�  X   sequencer�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  X   19ntr�  X   NNr�  X   B-NPr�  �r�  j�  j�  h,�r�  X   onr�  X   INr�  X   B-PPr�  �r�  X   yourr�  X   PRP$r�  X   B-NPr�  �r�  X   targetr�  X   NNr�  X   I-NPr�  �r�  X   gener�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  e]r�  (X   Extractr�  X   NNr�  X   B-NPr�  �r�  X   2ntr�  X   NNr�  X   I-NPr�  �r�  X   ofr�  X   INr�  X   B-PPr�  �r�  X   upstreamr�  X   JJr�  X   B-NPr�  �r�  j�  X   SYMr�  X   I-NPr�  �r�  X   19ntr�  X   CDr�  X   I-NPr�  �r�  X   siRNAr�  X   NNr�  X   I-NPr�  �r�  j�  X   SYMr�  X   B-NPr�  �r�  X   1ntr�  X   NNr�  X   I-NPr�  �r�  X   ofr�  X   INr�  X   B-PPr�  �r�  X
   downstreamr�  X   NNr�  X   B-NPr�  �r�  j�  j�  h,�r�  X   whichr�  X   WDTr�  X   B-NPr�  �r�  X   isr�  X   VBZr�  X   B-VPr�  �r�  X   totallyr�  X   RBr�  X   B-ADJPr�  �r�  X   22ntr�  X   JJr�  X   I-ADJPr�  �r�  X   ofr�  X   INr�  X   B-PPr�  �r�  X   nucleotidesr�  X   NNSr�  X   B-NPr�  �r�  j�  j�  h,�r�  e]r�  (X	   Antisenser�  X   JJr�  X   B-NPr�  �r�  X   shRNAr�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  X   Insertr�  X   VBr�  X   B-VPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   reverser�  X   JJr�  X   I-NPr�  �r�  X
   complementr�  X   NNr   X   I-NPr  �r  X   ofr  X   INr  X   B-PPr  �r  X   ther  X   DTr  X   B-NPr	  �r
  X   22ntr  X   NNr  X   I-NPr  �r  X   intor  X   INr  X   B-PPr  �r  X   shRNAr  X   NNr  X   B-NPr  �r  X   backboner  X   NNr  X   I-NPr  �r  j�  j�  h,�r  e]r  (X   Senser  X   JJr  X   B-NPr  �r   X   shRNAr!  X   NNr"  X   I-NPr#  �r$  j�  j�  h,�r%  X   Insertr&  X   VBr'  X   B-VPr(  �r)  X   ther*  X   DTr+  X   B-NPr,  �r-  X   22ntr.  X   NNr/  X   I-NPr0  �r1  j�  j�  h,�r2  X   fromr3  X   INr4  X   B-PPr5  �r6  X   stepr7  X   NNr8  X   B-NPr9  �r:  jN  X   CDr;  X   I-NPr<  �r=  j�  j�  h,�r>  X   intor?  X   INr@  X   B-PPrA  �rB  X   shRNArC  X   NNrD  X   B-NPrE  �rF  X   backbonerG  X   NNrH  X   I-NPrI  �rJ  j�  j�  h,�rK  e]rL  (X   ReplacerM  X   VBrN  X   B-VPrO  �rP  X   therQ  X   DTrR  X   B-NPrS  �rT  X   1strU  X   JJrV  X   I-NPrW  �rX  X   ntrY  X   NNrZ  X   I-NPr[  �r\  X   withr]  X   INr^  X   B-PPr_  �r`  X   otherra  X   JJrb  X   B-NPrc  �rd  X
   nucleotidere  X   NNrf  X   I-NPrg  �rh  j�  j�  h,�ri  X   replacerj  X   NNrk  X   B-NPrl  �rm  hX   NNrn  X   I-NPro  �rp  X   withrq  X   INrr  X   B-PPrs  �rt  h$X   NNru  X   B-NPrv  �rw  X   andrx  X   CCry  X   I-NPrz  �r{  X   vicer|  X   NNr}  X   I-NPr~  �r  X   versar�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  X   replacer�  X   NNr�  X   B-NPr�  �r�  h X   NNr�  X   I-NPr�  �r�  X   withr�  X   INr�  X   B-PPr�  �r�  h1X   NNr�  X   B-NPr�  �r�  X   andr�  X   CCr�  X   I-NPr�  �r�  X   vicer�  X   NNr�  X   I-NPr�  �r�  X   versar�  X   NNr�  X   I-NPr�  �r�  j�  j�  h,�r�  j�  j�  h,�r�  e]r�  (X   Nowr�  X   RBr�  X   B-ADVPr�  �r�  X   yourr�  X   PRP$r�  X   B-NPr�  �r�  X   shRNAr�  X   NNr�  X   I-NPr�  �r�  X	   designingr�  X   VBGr�  X   B-VPr�  �r�  X   jobr�  X   NNr�  X   B-NPr�  �r�  X   isr�  X   VBZr�  X   B-VPr�  �r�  X   doner�  X   VBNr�  X   I-VPr�  �r�  j�  j�  h,�r�  e]r�  (X   Your�  X   PRPr�  X   B-NPr�  �r�  X   seer�  X   VBPr�  X   B-VPr�  �r�  j�  j�  h,�r�  X   howr�  X   WRBr�  X   B-ADJPr�  �r�  X   easyr�  X   JJr�  X   I-ADJPr�  �r�  X   itr�  X   PRPr�  X   B-NPr�  �r�  X   isr�  X   VBZr�  X   B-VPr�  �r�  j�  j�  h,�r�  e]r�  (X	   Synthesisr�  X   NNr�  X   B-NPr�  �r�  X   ther�  X   DTr�  X   B-NPr�  �r�  X   shRNAr�  X   NNr�  X   I-NPr�  �r�  X   nucleotidesr�  X   NNSr�  X   I-NPr�  �r�  X   andr�  X   CCr�  h,�r�  X   insertr�  X   VBr�  X   B-VPr�  �r�  X   themr�  X   PRPr�  X   B-NPr�  �r�  X   intor�  X   INr�  X   B-PPr�  �r�  X   pTRIPZr�  X   NNr�  X   B-NPr�  �r�  X   vectorr�  X   NNr�  X   I-NPr   �r  j�  j�  h,�r  e]r  (X   Your  X   PRPr  X   B-NPr  �r  X   mayr  X   MDr	  X   B-VPr
  �r  X   alsor  X   RBr  X   I-VPr  �r  X   translocater  X   VBr  X   I-VPr  �r  X   ther  X   DTr  X   B-NPr  �r  X   shRNAr  X   NNr  X   I-NPr  �r  X   fromr  X   INr  X   B-PPr  �r  X   pTRIPZr   X   NNr!  X   B-NPr"  �r#  X   vectorr$  X   NNr%  X   I-NPr&  �r'  X   intor(  X   INr)  X   B-PPr*  �r+  X   pGIPZr,  X   NNr-  X   B-NPr.  �r/  X   vectorr0  X   NNr1  X   I-NPr2  �r3  j�  j�  h,�r4  eeX
   conll_depsr5  ]r6  (XS  1	Anchor/IN	_	B	B	_	5	nsubj	_	_
2	the/DT	_	B	B	_	1	dep	_	_
3	siRNA/NN	_	I	I	_	5	dep	_	_
4	sequence/NN	_	I	I	_	5	dep	_	_
5	(/(	_	O	O	_	0	root	_	_
6	19nt/NN	_	B	B	_	5	dep	_	_
7	)/)	_	O	O	_	5	dep	_	_
8	on/IN	_	B	B	_	9	dep	_	_
9	your/PRP$	_	B	B	_	5	nmod	_	_
10	target/NN	_	I	I	_	9	dep	_	_
11	gene/NN	_	I	I	_	9	dep	_	_
12	./.	_	O	O	_	5	dep	_	_
r7  X8  1	Extract/NN	_	B	B	_	2	dep	_	_
2	2nt/NN	_	I	I	_	0	root	_	_
3	of/IN	_	B	B	_	15	dep	_	_
4	upstream/JJ	_	B	B	_	15	dep	_	_
5	+/SYM	_	I	I	_	7	dep	_	_
6	19nt/CD	_	I	I	_	7	dep	_	_
7	siRNA/NN	_	I	I	_	15	nummod	_	_
8	+/SYM	_	B	B	_	15	dep	_	_
9	1nt/NN	_	I	I	_	11	dep	_	_
10	of/IN	_	B	B	_	11	dep	_	_
11	downstream/NN	_	B	B	_	15	nummod	_	_
12	,/,	_	O	O	_	15	dep	_	_
13	which/WDT	_	B	B	_	15	dep	_	_
14	is/VBZ	_	B	B	_	15	dep	_	_
15	totally/RB	_	B	B	_	2	dep	_	_
16	22nt/JJ	_	I	I	_	15	dep	_	_
17	of/IN	_	B	B	_	15	dep	_	_
18	nucleotides/NNS	_	B	B	_	15	dep	_	_
19	./.	_	O	O	_	2	dep	_	_
r8  X�  1	Antisense/JJ	_	B	B	_	0	root	_	_
2	shRNA/NN	_	I	I	_	1	dep	_	_
3	:/:	_	O	O	_	1	dep	_	_
4	Insert/VB	_	B	B	_	1	dep	_	_
5	the/DT	_	B	B	_	1	dep	_	_
6	reverse/JJ	_	I	I	_	1	dep	_	_
7	complement/NN	_	I	I	_	1	dep	_	_
8	of/IN	_	B	B	_	1	dep	_	_
9	the/DT	_	B	B	_	1	dep	_	_
10	22nt/NN	_	I	I	_	1	dep	_	_
11	into/IN	_	B	B	_	1	dep	_	_
12	shRNA/NN	_	B	B	_	1	dep	_	_
13	backbone/NN	_	I	I	_	1	dep	_	_
14	./.	_	O	O	_	1	dep	_	_
r9  X�  1	Sense/JJ	_	B	B	_	2	dep	_	_
2	shRNA/NN	_	I	I	_	0	root	_	_
3	:/:	_	O	O	_	11	dep	_	_
4	Insert/VB	_	B	B	_	11	dep	_	_
5	the/DT	_	B	B	_	7	dep	_	_
6	22nt/NN	_	I	I	_	7	dep	_	_
7	(/(	_	O	O	_	11	nummod	_	_
8	from/IN	_	B	B	_	11	dep	_	_
9	step/NN	_	B	B	_	11	dep	_	_
10	2/CD	_	I	I	_	11	dep	_	_
11	)/)	_	O	O	_	2	dep	_	_
12	into/IN	_	B	B	_	11	dep	_	_
13	shRNA/NN	_	B	B	_	11	dep	_	_
14	backbone/NN	_	I	I	_	11	dep	_	_
15	./.	_	O	O	_	2	dep	_	_
r:  X�  1	Replace/VB	_	B	B	_	3	nsubj	_	_
2	the/DT	_	B	B	_	3	dep	_	_
3	1st/JJ	_	I	I	_	0	root	_	_
4	nt/NN	_	I	I	_	3	dep	_	_
5	with/IN	_	B	B	_	3	dep	_	_
6	other/JJ	_	B	B	_	3	dep	_	_
7	nucleotide/NN	_	I	I	_	3	dep	_	_
8	(/(	_	O	O	_	3	dep	_	_
9	replace/NN	_	B	B	_	3	dep	_	_
10	A/NN	_	I	I	_	3	dep	_	_
11	with/IN	_	B	B	_	3	dep	_	_
12	G/NN	_	B	B	_	3	dep	_	_
13	and/CC	_	I	I	_	3	dep	_	_
14	vice/NN	_	I	I	_	3	dep	_	_
15	versa/NN	_	I	I	_	3	dep	_	_
16	;/:	_	O	O	_	3	dep	_	_
17	replace/NN	_	B	B	_	3	dep	_	_
18	C/NN	_	I	I	_	3	dep	_	_
19	with/IN	_	B	B	_	3	dep	_	_
20	T/NN	_	B	B	_	3	dep	_	_
21	and/CC	_	I	I	_	3	dep	_	_
22	vice/NN	_	I	I	_	3	dep	_	_
23	versa/NN	_	I	I	_	3	dep	_	_
24	)/)	_	O	O	_	3	dep	_	_
25	./.	_	O	O	_	3	dep	_	_
r;  X�   1	Now/RB	_	B	B	_	5	nsubj	_	_
2	your/PRP$	_	B	B	_	1	dep	_	_
3	shRNA/NN	_	I	I	_	5	dep	_	_
4	designing/VBG	_	B	B	_	5	dep	_	_
5	job/NN	_	B	B	_	0	root	_	_
6	is/VBZ	_	B	B	_	5	dep	_	_
7	done/VBN	_	I	I	_	5	dep	_	_
8	./.	_	O	O	_	5	dep	_	_
r<  X�   1	You/PRP	_	B	B	_	5	nsubj	_	_
2	see/VBP	_	B	B	_	1	dep	_	_
3	,/,	_	O	O	_	5	dep	_	_
4	how/WRB	_	B	B	_	5	dep	_	_
5	easy/JJ	_	I	I	_	0	root	_	_
6	it/PRP	_	B	B	_	5	dep	_	_
7	is/VBZ	_	B	B	_	5	dep	_	_
8	!/.	_	O	O	_	5	dep	_	_
r=  XI  1	Synthesis/NN	_	B	B	_	4	nsubj	_	_
2	the/DT	_	B	B	_	4	dep	_	_
3	shRNA/NN	_	I	I	_	4	dep	_	_
4	nucleotides/NNS	_	I	I	_	0	root	_	_
5	and/CC	_	O	O	_	4	dep	_	_
6	insert/VB	_	B	B	_	4	dep	_	_
7	them/PRP	_	B	B	_	8	dep	_	_
8	into/IN	_	B	B	_	4	nmod	_	_
9	pTRIPZ/NN	_	B	B	_	8	dep	_	_
10	vector/NN	_	I	I	_	8	dep	_	_
11	./.	_	O	O	_	4	dep	_	_
r>  X}  1	You/PRP	_	B	B	_	2	dep	_	_
2	may/MD	_	B	B	_	0	root	_	_
3	also/RB	_	I	I	_	2	dep	_	_
4	translocate/VB	_	I	I	_	3	dep	_	_
5	the/DT	_	B	B	_	3	dep	_	_
6	shRNA/NN	_	I	I	_	3	dep	_	_
7	from/IN	_	B	B	_	9	dep	_	_
8	pTRIPZ/NN	_	B	B	_	9	dep	_	_
9	vector/NN	_	I	I	_	3	nummod	_	_
10	into/IN	_	B	B	_	3	dep	_	_
11	pGIPZ/NN	_	B	B	_	3	dep	_	_
12	vector/NN	_	I	I	_	3	dep	_	_
13	./.	_	O	O	_	2	dep	_	_
r?  eX   parse_treesr@  ]rA  (cnltk.tree
Tree
rB  )�rC  jB  )�rD  (jB  )�rE  jB  )�rF  X   AnchorrG  a}rH  X   _labelrI  X   NNPrJ  sba}rK  jI  X   NPrL  sbjB  )�rM  (jB  )�rN  (jB  )�rO  (jB  )�rP  X   therQ  a}rR  jI  X   DTrS  sbjB  )�rT  X   siRNArU  a}rV  jI  X   NNrW  sbjB  )�rX  X   sequencerY  a}rZ  jI  X   NNr[  sbe}r\  jI  X   NPr]  sbjB  )�r^  (jB  )�r_  X   -LRB-r`  a}ra  jI  X   -LRB-rb  sbjB  )�rc  jB  )�rd  X   19ntre  a}rf  jI  X   NNPrg  sba}rh  jI  X   NPri  sbjB  )�rj  X   -RRB-rk  a}rl  jI  X   -RRB-rm  sbe}rn  jI  X   PRNro  sbe}rp  jI  X   NPrq  sbjB  )�rr  (jB  )�rs  X   onrt  a}ru  jI  X   INrv  sbjB  )�rw  (jB  )�rx  X   yourry  a}rz  jI  X   PRP$r{  sbjB  )�r|  X   targetr}  a}r~  jI  X   NNr  sbjB  )�r�  X   gener�  a}r�  jI  X   NNr�  sbe}r�  jI  X   NPr�  sbe}r�  jI  X   PPr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  j�  a}r�  jI  j�  sbe}r�  jI  X   NPr�  sba}r�  jI  X   ROOTr�  sbjB  )�r�  jB  )�r�  (jB  )�r�  (jB  )�r�  (jB  )�r�  X   Extractr�  a}r�  jI  X   JJr�  sbjB  )�r�  X   2ntr�  a}r�  jI  X   NNSr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r�  X   ofr�  a}r�  jI  X   INr�  sbjB  )�r�  (jB  )�r�  X   upstreamr�  a}r�  jI  X   JJr�  sbjB  )�r�  X   +19r�  a}r�  jI  X   CDr�  sbe}r�  jI  X   NPr�  sbe}r�  jI  X   PPr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r�  X   ntr�  a}r�  jI  X   VBZr�  sbjB  )�r�  (jB  )�r�  (jB  )�r�  X   siRNAr�  a}r�  jI  X   JJr�  sbjB  )�r�  j�  a}r�  jI  X   JJr�  sbjB  )�r�  X   1ntr�  a}r�  jI  X   NNSr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r�  X   ofr�  a}r�  jI  X   INr�  sbjB  )�r�  (jB  )�r�  jB  )�r�  X
   downstreamr�  a}r�  jI  X   JJr�  sba}r�  jI  X   NPr�  sbjB  )�r�  j�  a}r�  jI  j�  sbjB  )�r�  (jB  )�r�  jB  )�r�  X   whichr�  a}r�  jI  X   WDTr�  sba}r�  jI  X   WHNPr�  sbjB  )�r�  jB  )�r�  (jB  )�r�  X   isr�  a}r�  jI  X   VBZr�  sbjB  )�r�  (jB  )�r�  X   totallyr�  a}r�  jI  X   RBr�  sbjB  )�r�  X   22ntr�  a}r�  jI  X   JJr�  sbjB  )�r�  (jB  )�r�  X   ofr�  a}r�  jI  X   INr�  sbjB  )�r�  jB  )�r�  X   nucleotidesr�  a}r�  jI  X   NNSr�  sba}r�  jI  X   NPr�  sbe}r�  jI  X   PPr�  sbe}r�  jI  X   ADJPr�  sbe}r�  jI  X   VPr�  sba}r�  jI  h0sbe}r�  jI  X   SBARr   sbe}r  jI  X   NPr  sbe}r  jI  X   PPr  sbe}r  jI  X   NPr  sbe}r  jI  X   VPr  sbjB  )�r	  j�  a}r
  jI  j�  sbe}r  jI  h0sba}r  jI  X   ROOTr  sbjB  )�r  jB  )�r  (jB  )�r  (jB  )�r  X	   Antisenser  a}r  jI  X   JJr  sbjB  )�r  X   shRNAr  a}r  jI  X   NNr  sbe}r  jI  X   NPr  sbjB  )�r  j�  a}r  jI  j�  sbjB  )�r  (jB  )�r  jB  )�r  X   Insertr   a}r!  jI  X   NNPr"  sba}r#  jI  X   NPr$  sbjB  )�r%  (jB  )�r&  (jB  )�r'  X   ther(  a}r)  jI  X   DTr*  sbjB  )�r+  X   reverser,  a}r-  jI  X   JJr.  sbjB  )�r/  X
   complementr0  a}r1  jI  X   NNr2  sbe}r3  jI  X   NPr4  sbjB  )�r5  (jB  )�r6  X   ofr7  a}r8  jI  X   INr9  sbjB  )�r:  (jB  )�r;  X   ther<  a}r=  jI  X   DTr>  sbjB  )�r?  (jB  )�r@  X   22ntrA  a}rB  jI  X   JJrC  sbjB  )�rD  (jB  )�rE  X   intorF  a}rG  jI  X   INrH  sbjB  )�rI  jB  )�rJ  X   shRNArK  a}rL  jI  X   NNrM  sba}rN  jI  X   NPrO  sbe}rP  jI  X   PPrQ  sbe}rR  jI  X   ADJPrS  sbjB  )�rT  X   backbonerU  a}rV  jI  X   NNrW  sbe}rX  jI  X   NPrY  sbe}rZ  jI  X   PPr[  sbe}r\  jI  X   NPr]  sbe}r^  jI  X   NPr_  sbjB  )�r`  j�  a}ra  jI  j�  sbe}rb  jI  X   NPrc  sba}rd  jI  X   ROOTre  sbjB  )�rf  jB  )�rg  (jB  )�rh  (jB  )�ri  X   Senserj  a}rk  jI  X   NNrl  sbjB  )�rm  X   shRNArn  a}ro  jI  X   NNrp  sbe}rq  jI  X   NPrr  sbjB  )�rs  j�  a}rt  jI  j�  sbjB  )�ru  (jB  )�rv  jB  )�rw  X   Insertrx  a}ry  jI  X   NNPrz  sba}r{  jI  X   NPr|  sbjB  )�r}  (jB  )�r~  X   ther  a}r�  jI  X   DTr�  sbjB  )�r�  (jB  )�r�  (jB  )�r�  jB  )�r�  X   22ntr�  a}r�  jI  X   JJr�  sba}r�  jI  X   ADJPr�  sbjB  )�r�  (jB  )�r�  X   -LRB-r�  a}r�  jI  X   -LRB-r�  sbjB  )�r�  (jB  )�r�  X   fromr�  a}r�  jI  X   INr�  sbjB  )�r�  (jB  )�r�  X   stepr�  a}r�  jI  X   NNr�  sbjB  )�r�  jN  a}r�  jI  X   CDr�  sbe}r�  jI  X   NPr�  sbe}r�  jI  X   PPr�  sbjB  )�r�  X   -RRB-r�  a}r�  jI  X   -RRB-r�  sbe}r�  jI  X   PRNr�  sbe}r�  jI  X   ADJPr�  sbjB  )�r�  (jB  )�r�  X   intor�  a}r�  jI  X   INr�  sbjB  )�r�  jB  )�r�  X   shRNAr�  a}r�  jI  X   NNr�  sba}r�  jI  X   NPr�  sbe}r�  jI  X   PPr�  sbe}r�  jI  X   ADJPr�  sbjB  )�r�  X   backboner�  a}r�  jI  X   NNr�  sbe}r�  jI  X   NPr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  j�  a}r�  jI  j�  sbe}r�  jI  X   NPr�  sba}r�  jI  X   ROOTr�  sbjB  )�r�  jB  )�r�  (jB  )�r�  (jB  )�r�  X   Replacer�  a}r�  jI  X   VBr�  sbjB  )�r�  (jB  )�r�  (jB  )�r�  X   ther�  a}r�  jI  X   DTr�  sbjB  )�r�  X   1str�  a}r�  jI  X   JJr�  sbjB  )�r�  X   ntr�  a}r�  jI  X   NNr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r�  X   withr�  a}r�  jI  X   INr�  sbjB  )�r�  (jB  )�r�  X   otherr�  a}r�  jI  X   JJr�  sbjB  )�r�  X
   nucleotider�  a}r�  jI  X   NNr�  sbjB  )�r�  (jB  )�r�  X   -LRB-r�  a}r�  jI  X   -LRB-r�  sbjB  )�r�  jB  )�r�  (jB  )�r�  X   replacer�  a}r�  jI  X   VBr�  sbjB  )�r�  (jB  )�r�  (jB  )�r�  jB  )�r�  ha}r�  jI  X   NNPr�  sba}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r   X   withr  a}r  jI  X   INr  sbjB  )�r  (jB  )�r  jB  )�r  h$a}r  jI  X   NNPr  sba}r	  jI  X   NPr
  sbjB  )�r  X   andr  a}r  jI  X   CCr  sbjB  )�r  (jB  )�r  X   vicer  a}r  jI  X   NNr  sbjB  )�r  X   versar  a}r  jI  X   NNSr  sbe}r  jI  X   NPr  sbe}r  jI  X   NPr  sbe}r  jI  X   PPr  sbe}r  jI  X   NPr  sbjB  )�r   j�  a}r!  jI  j�  sbjB  )�r"  (jB  )�r#  jB  )�r$  (jB  )�r%  X   replacer&  a}r'  jI  X   VBr(  sbjB  )�r)  (jB  )�r*  jB  )�r+  h a}r,  jI  X   NNPr-  sba}r.  jI  X   NPr/  sbjB  )�r0  (jB  )�r1  X   withr2  a}r3  jI  X   INr4  sbjB  )�r5  jB  )�r6  h1a}r7  jI  X   NNPr8  sba}r9  jI  X   NPr:  sbe}r;  jI  X   PPr<  sbe}r=  jI  X   NPr>  sbe}r?  jI  X   VPr@  sba}rA  jI  h0sbjB  )�rB  X   andrC  a}rD  jI  X   CCrE  sbjB  )�rF  (jB  )�rG  X   vicerH  a}rI  jI  X   RBrJ  sbjB  )�rK  X   versarL  a}rM  jI  X   RBrN  sbe}rO  jI  X   ADVPrP  sbe}rQ  jI  X   UCPrR  sbe}rS  jI  X   NPrT  sbe}rU  jI  X   VPrV  sba}rW  jI  h0sbjB  )�rX  X   -RRB-rY  a}rZ  jI  X   -RRB-r[  sbe}r\  jI  X   PRNr]  sbe}r^  jI  X   NPr_  sbe}r`  jI  X   PPra  sbe}rb  jI  X   NPrc  sbe}rd  jI  X   VPre  sbjB  )�rf  j�  a}rg  jI  j�  sbe}rh  jI  h0sba}ri  jI  X   ROOTrj  sbjB  )�rk  jB  )�rl  (jB  )�rm  jB  )�rn  X   Nowro  a}rp  jI  X   RBrq  sba}rr  jI  X   ADVPrs  sbjB  )�rt  (jB  )�ru  (jB  )�rv  X   yourrw  a}rx  jI  X   PRP$ry  sbjB  )�rz  X   shRNAr{  a}r|  jI  X   NNr}  sbe}r~  jI  X   NPr  sbjB  )�r�  (jB  )�r�  X	   designingr�  a}r�  jI  X   VBGr�  sbjB  )�r�  jB  )�r�  X   jobr�  a}r�  jI  X   NNr�  sba}r�  jI  X   NPr�  sbe}r�  jI  X   VPr�  sbe}r�  jI  X   NPr�  sbjB  )�r�  (jB  )�r�  X   isr�  a}r�  jI  X   VBZr�  sbjB  )�r�  jB  )�r�  X   doner�  a}r�  jI  X   VBNr�  sba}r�  jI  X   VPr�  sbe}r�  jI  X   VPr�  sbjB  )�r�  j�  a}r�  jI  j�  sbe}r�  jI  h0sba}r�  jI  X   ROOTr�  sbjB  )�r�  jB  )�r�  (jB  )�r�  (jB  )�r�  jB  )�r�  X   Your�  a}r�  jI  X   PRPr�  sba}r�  jI  X   NPr�  sbjB  )�r�  jB  )�r�  X   seer�  a}r�  jI  X   VBPr�  sba}r�  jI  X   VPr�  sbe}r�  jI  h0sbjB  )�r�  j�  a}r�  jI  j�  sbjB  )�r�  (jB  )�r�  (jB  )�r�  X   howr�  a}r�  jI  X   WRBr�  sbjB  )�r�  jB  )�r�  X   easyr�  a}r�  jI  X   JJr�  sba}r�  jI  X   ADJPr�  sbe}r�  jI  X   WHADVPr�  sbjB  )�r�  (jB  )�r�  jB  )�r�  X   itr�  a}r�  jI  X   PRPr�  sba}r�  jI  X   NPr�  sbjB  )�r�  jB  )�r�  X   isr�  a}r�  jI  X   VBZr�  sba}r�  jI  X   VPr�  sbe}r�  jI  h0sbe}r�  jI  X   SBARr�  sbjB  )�r�  j�  a}r�  jI  j�  sbe}r�  jI  X   FRAGr�  sba}r�  jI  X   ROOTr�  sbjB  )�r�  jB  )�r�  (jB  )�r�  (jB  )�r�  (jB  )�r�  X	   Synthesisr�  a}r�  jI  X   VBr�  sbjB  )�r�  (jB  )�r�  X   ther�  a}r�  jI  X   DTr�  sbjB  )�r�  X   shRNAr�  a}r�  jI  X   NNr�  sbjB  )�r�  X   nucleotidesr�  a}r�  jI  X   NNSr�  sbe}r�  jI  X   NPr�  sbe}r�  jI  X   VPr�  sbjB  )�r�  X   andr�  a}r�  jI  X   CCr�  sbjB  )�r�  (jB  )�r�  X   insertr�  a}r�  jI  X   VBr�  sbjB  )�r 	  jB  )�r	  X   themr	  a}r	  jI  X   PRPr	  sba}r	  jI  X   NPr	  sbjB  )�r	  (jB  )�r	  X   intor		  a}r
	  jI  X   INr	  sbjB  )�r	  (jB  )�r	  X   pTRIPZr	  a}r	  jI  X   NNr	  sbjB  )�r	  X   vectorr	  a}r	  jI  X   NNr	  sbe}r	  jI  X   NPr	  sbe}r	  jI  X   PPr	  sbe}r	  jI  X   VPr	  sbe}r	  jI  X   VPr	  sbjB  )�r	  j�  a}r	  jI  j�  sbe}r	  jI  h0sba}r 	  jI  X   ROOTr!	  sbjB  )�r"	  jB  )�r#	  (jB  )�r$	  jB  )�r%	  X   Your&	  a}r'	  jI  X   PRPr(	  sba}r)	  jI  X   NPr*	  sbjB  )�r+	  (jB  )�r,	  X   mayr-	  a}r.	  jI  X   MDr/	  sbjB  )�r0	  jB  )�r1	  X   alsor2	  a}r3	  jI  X   RBr4	  sba}r5	  jI  X   ADVPr6	  sbjB  )�r7	  (jB  )�r8	  X   translocater9	  a}r:	  jI  X   VBr;	  sbjB  )�r<	  (jB  )�r=	  X   ther>	  a}r?	  jI  X   DTr@	  sbjB  )�rA	  X   shRNArB	  a}rC	  jI  X   NNrD	  sbe}rE	  jI  X   NPrF	  sbjB  )�rG	  (jB  )�rH	  X   fromrI	  a}rJ	  jI  X   INrK	  sbjB  )�rL	  (jB  )�rM	  X   pTRIPZrN	  a}rO	  jI  X   NNrP	  sbjB  )�rQ	  X   vectorrR	  a}rS	  jI  X   NNrT	  sbe}rU	  jI  X   NPrV	  sbe}rW	  jI  X   PPrX	  sbjB  )�rY	  (jB  )�rZ	  X   intor[	  a}r\	  jI  X   INr]	  sbjB  )�r^	  (jB  )�r_	  X   pGIPZr`	  a}ra	  jI  X   NNrb	  sbjB  )�rc	  X   vectorrd	  a}re	  jI  X   NNrf	  sbe}rg	  jI  X   NPrh	  sbe}ri	  jI  X   PPrj	  sbe}rk	  jI  X   VPrl	  sbe}rm	  jI  X   VPrn	  sbjB  )�ro	  j�  a}rp	  jI  j�  sbe}rq	  jI  h0sba}rr	  jI  X   ROOTrs	  sbeubj0  h�X   arg1_tagrt	  h�X   arg2_tagru	  h�X
   parse_treerv	  jC  j3  Nubh)�rw	  }rx	  (hK hK K�ry	  hK	K�rz	  h	hj0  h�jt	  h�ju	  j  jv	  jC  j3  Nubh)�r{	  }r|	  (hKhK K�r}	  hKK�r~	  h	hj0  j  jt	  j  ju	  j  jv	  j�  j3  Nubh)�r	  }r�	  (hKhK K�r�	  hK
K�r�	  h	hj0  j  jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj0  j  jt	  j  ju	  j#  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j'  jt	  j-  ju	  j4  jv	  j  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j7  jt	  j-  ju	  j>  jv	  j  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK
�r�	  h	hj0  jB  jt	  jH  ju	  jP  jv	  jf  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  jS  jt	  jH  ju	  jZ  jv	  jf  j3  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj0  j^  jt	  jd  ju	  jk  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj0  jn  jt	  jd  ju	  ju  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj0  jy  jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK
�r�	  h	hj0  j�  jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhK
K�r�	  hKK	�r�	  h	hj0  j�  jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK
�r�	  hKK�r�	  h	hj0  j�  jt	  jP  ju	  j�  jv	  jf  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hK hKK�r�	  hKK�r�	  h	hj0  j�  jt	  h�ju	  j�  jv	  jC  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j�  jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hK	K
�r�	  h	hj0  j�  jt	  j4  ju	  j�  jv	  j  j3  Nubh)�r�	  }r�	  (hKhKK	�r�	  hK	K
�r�	  h	hj0  j�  jt	  j  ju	  j	  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK	�r�	  hKK�r�	  h	hj0  j  jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK	�r�	  hKK�r�	  h	hj0  j  jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j!  jt	  j'  ju	  j,  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j/  jt	  j'  ju	  j4  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  j8  jt	  j'  ju	  j@  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  jD  jt	  jJ  ju	  jP  jv	  j�  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK�r�	  h	hj0  jT  jt	  jZ  ju	  j`  jv	  j"	  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hKK	�r�	  h	hj0  jc  jt	  jZ  ju	  jj  jv	  j"	  j3  Nubh)�r�	  }r�	  (hKhKK�r�	  hK
K�r�	  h	hj0  jm  jt	  jZ  ju	  jt  jv	  j"	  j3  Nubh)�r�	  }r�	  (hK hK K�r�	  hKK�r�	  h	hj0  h,jt	  h�ju	  j�  jv	  jC  j3  Nubh)�r�	  }r�	  (hKhK K�r�	  hKK�r�	  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�	  }r 
  (hKhK K�r
  hKK	�r
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r
  }r
  (hKhK K�r
  hKK�r
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r
  }r
  (hKhK K�r	
  hKK�r

  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r
  }r
  (hKhK K�r
  hKK�r
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r
  }r
  (hKhK K�r
  hKK�r
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r
  }r
  (hKhKK�r
  hK K�r
  h	hj0  h,jt	  j-  ju	  j  jv	  j  j3  Nubh)�r
  }r
  (hKhKK�r
  hK	K
�r
  h	hj0  h,jt	  j-  ju	  j�  jv	  j  j3  Nubh)�r
  }r
  (hKhKK�r
  hKK�r
  h	hj0  h,jt	  jH  ju	  j�  jv	  jf  j3  Nubh)�r
  }r 
  (hKhKK�r!
  hK K�r"
  h	hj0  h,jt	  jH  ju	  j  jv	  jf  j3  Nubh)�r#
  }r$
  (hKhK K�r%
  hKK	�r&
  h	hj0  h,jt	  jd  ju	  j  jv	  j�  j3  Nubh)�r'
  }r(
  (hKhK K�r)
  hK	K
�r*
  h	hj0  h,jt	  jd  ju	  j	  jv	  j�  j3  Nubh)�r+
  }r,
  (hKhK K�r-
  hKK�r.
  h	hj0  h,jt	  jd  ju	  j  jv	  j�  j3  Nubh)�r/
  }r0
  (hKhK K�r1
  hKK�r2
  h	hj0  h,jt	  jd  ju	  j  jv	  j�  j3  Nubh)�r3
  }r4
  (hKhK K�r5
  hKK�r6
  h	hj0  h,jt	  jd  ju	  j'  jv	  j�  j3  Nubh)�r7
  }r8
  (hKhK K�r9
  hKK�r:
  h	hj0  h,jt	  jd  ju	  j,  jv	  j�  j3  Nubh)�r;
  }r<
  (hKhK K�r=
  hKK�r>
  h	hj0  h,jt	  jd  ju	  j4  jv	  j�  j3  Nubh)�r?
  }r@
  (hKhK K�rA
  hKK�rB
  h	hj0  h,jt	  jd  ju	  j@  jv	  j�  j3  Nubh)�rC
  }rD
  (hKhK K�rE
  hKK�rF
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�rG
  }rH
  (hKhK K�rI
  hKK�rJ
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�rK
  }rL
  (hKhK K�rM
  hKK
�rN
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�rO
  }rP
  (hKhKK�rQ
  hK K�rR
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rS
  }rT
  (hKhKK�rU
  hKK�rV
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rW
  }rX
  (hK hKK�rY
  hK K�rZ
  h	hj0  h,jt	  h�ju	  h�jv	  jC  j3  Nubh)�r[
  }r\
  (hK hKK�r]
  hK	K�r^
  h	hj0  h,jt	  h�ju	  j  jv	  jC  j3  Nubh)�r_
  }r`
  (hK hK	K�ra
  hK K�rb
  h	hj0  h,jt	  j  ju	  h�jv	  jC  j3  Nubh)�rc
  }rd
  (hK hK	K�re
  hKK�rf
  h	hj0  h,jt	  j  ju	  h�jv	  jC  j3  Nubh)�rg
  }rh
  (hK hK	K�ri
  hKK�rj
  h	hj0  h,jt	  j  ju	  j�  jv	  jC  j3  Nubh)�rk
  }rl
  (hKhKK�rm
  hK K�rn
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�ro
  }rp
  (hKhKK�rq
  hKK�rr
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rs
  }rt
  (hKhKK�ru
  hKK	�rv
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rw
  }rx
  (hKhKK�ry
  hK
K�rz
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r{
  }r|
  (hKhKK�r}
  hKK�r~
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j4  ju	  j-  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j4  ju	  j>  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj0  h,jt	  j4  ju	  j  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j>  ju	  j-  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j>  ju	  j4  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj0  h,jt	  j>  ju	  j  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hK	K
�r�
  h	hj0  h,jt	  j>  ju	  j�  jv	  j  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hK K�r�
  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK	�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hK
K�r�
  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j#  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hK K�r�
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hK
K�r�
  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhKK	�r�
  hKK�r�
  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhK
K�r�
  hK K�r�
  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhK
K�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhK
K�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r�
  }r�
  (hKhK
K�r�
  hKK�r�
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r�
  }r   (hKhK
K�r  hKK�r  h	hj0  h,jt	  j  ju	  j#  jv	  j�  j3  Nubh)�r  }r  (hKhK
K�r  hKK�r  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r  }r  (hKhK
K�r	  hKK�r
  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r  }r  (hKhK
K�r  hKK�r  h	hj0  h,jt	  j  ju	  j�  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hK K�r  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK	�r  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r  }r   (hKhKK�r!  hK
K�r"  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r#  }r$  (hKhKK�r%  hKK�r&  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�r'  }r(  (hKhKK�r)  hKK�r*  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r+  }r,  (hKhKK�r-  hKK�r.  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r/  }r0  (hKhKK�r1  hKK�r2  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r3  }r4  (hKhKK�r5  hK K�r6  h	hj0  h,jt	  j#  ju	  j  jv	  j�  j3  Nubh)�r7  }r8  (hKhKK�r9  hKK�r:  h	hj0  h,jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�r;  }r<  (hKhKK�r=  hKK�r>  h	hj0  h,jt	  j#  ju	  j  jv	  j�  j3  Nubh)�r?  }r@  (hKhKK�rA  hKK	�rB  h	hj0  h,jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�rC  }rD  (hKhKK�rE  hK
K�rF  h	hj0  h,jt	  j#  ju	  j  jv	  j�  j3  Nubh)�rG  }rH  (hKhKK�rI  hKK�rJ  h	hj0  h,jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�rK  }rL  (hKhKK�rM  hKK�rN  h	hj0  h,jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�rO  }rP  (hKhKK�rQ  hKK�rR  h	hj0  h,jt	  j#  ju	  j�  jv	  j�  j3  Nubh)�rS  }rT  (hKhKK�rU  hKK�rV  h	hj0  h,jt	  j�  ju	  jH  jv	  jf  j3  Nubh)�rW  }rX  (hKhKK�rY  hKK
�rZ  h	hj0  h,jt	  j�  ju	  jP  jv	  jf  j3  Nubh)�r[  }r\  (hKhKK�r]  hKK�r^  h	hj0  h,jt	  j�  ju	  jZ  jv	  jf  j3  Nubh)�r_  }r`  (hKhKK�ra  hK K�rb  h	hj0  h,jt	  j�  ju	  j  jv	  jf  j3  Nubh)�rc  }rd  (hKhKK
�re  hKK�rf  h	hj0  h,jt	  jP  ju	  jH  jv	  jf  j3  Nubh)�rg  }rh  (hKhKK
�ri  hKK�rj  h	hj0  h,jt	  jP  ju	  jZ  jv	  jf  j3  Nubh)�rk  }rl  (hKhKK
�rm  hK K�rn  h	hj0  h,jt	  jP  ju	  j  jv	  jf  j3  Nubh)�ro  }rp  (hKhKK�rq  hKK�rr  h	hj0  h,jt	  jZ  ju	  jH  jv	  jf  j3  Nubh)�rs  }rt  (hKhKK�ru  hKK�rv  h	hj0  h,jt	  jZ  ju	  j�  jv	  jf  j3  Nubh)�rw  }rx  (hKhKK�ry  hKK
�rz  h	hj0  h,jt	  jZ  ju	  jP  jv	  jf  j3  Nubh)�r{  }r|  (hKhKK�r}  hK K�r~  h	hj0  h,jt	  jZ  ju	  j  jv	  jf  j3  Nubh)�r  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  ju  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  jk  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj0  h,jt	  ju  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK	K
�r�  h	hj0  h,jt	  ju  ju	  j	  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j,  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j4  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  ju  ju	  j@  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  jk  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  ju  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj0  h,jt	  jk  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK	K
�r�  h	hj0  h,jt	  jk  ju	  j	  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j,  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j4  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jk  ju	  j@  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK
�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK
�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK
�r�  hK K�r�  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhK K �r�  hK K �r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j"	  j3  Nubh)�r�  }r   (hKhK K �r  hKK�r  h	hj0  h,jt	  j�  ju	  jZ  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hKK�r  h	hj0  h,jt	  j�  ju	  j`  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r	  hKK	�r
  h	hj0  h,jt	  j�  ju	  jj  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hK
K�r  h	hj0  h,jt	  j�  ju	  jt  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hK K �r  h	hj0  h,jt	  j�  ju	  j�  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hKK�r  h	hj0  h,jt	  j�  ju	  jZ  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hKK�r  h	hj0  h,jt	  j�  ju	  j`  jv	  j"	  j3  Nubh)�r  }r  (hKhK K �r  hKK	�r  h	hj0  h,jt	  j�  ju	  jj  jv	  j"	  j3  Nubh)�r  }r   (hKhK K �r!  hK
K�r"  h	hj0  h,jt	  j�  ju	  jt  jv	  j"	  j3  Nubh)�r#  }r$  (hK hKK�r%  hK K�r&  h	hj0  h,jt	  j�  ju	  h�jv	  jC  j3  Nubh)�r'  }r(  (hK hKK�r)  hKK�r*  h	hj0  h,jt	  j�  ju	  h�jv	  jC  j3  Nubh)�r+  }r,  (hK hKK�r-  hK	K�r.  h	hj0  h,jt	  j�  ju	  j  jv	  jC  j3  Nubh)�r/  }r0  (hKhKK�r1  hK K�r2  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r3  }r4  (hKhKK�r5  hKK�r6  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r7  }r8  (hKhKK�r9  hKK�r:  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r;  }r<  (hKhKK�r=  hKK	�r>  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r?  }r@  (hKhKK�rA  hK
K�rB  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rC  }rD  (hKhKK�rE  hKK�rF  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rG  }rH  (hKhKK�rI  hKK�rJ  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�rK  }rL  (hKhKK�rM  hKK�rN  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rO  }rP  (hKhKK�rQ  hK K�rR  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rS  }rT  (hKhKK�rU  hKK�rV  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rW  }rX  (hKhKK�rY  hKK�rZ  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r[  }r\  (hKhKK�r]  hKK	�r^  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r_  }r`  (hKhKK�ra  hK
K�rb  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rc  }rd  (hKhKK�re  hKK�rf  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rg  }rh  (hKhKK�ri  hKK�rj  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�rk  }rl  (hKhKK�rm  hKK�rn  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�ro  }rp  (hKhKK�rq  hK K�rr  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�rs  }rt  (hKhKK�ru  hKK�rv  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�rw  }rx  (hKhKK�ry  hKK�rz  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r{  }r|  (hKhKK�r}  hKK	�r~  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r  }r�  (hKhKK�r�  hK
K�r�  h	hj0  h,jt	  j�  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j#  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j�  jv	  j�  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j-  jv	  j  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j4  jv	  j  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j>  jv	  j  j3  Nubh)�r�  }r�  (hKhK K�r�  hK	K
�r�  h	hj0  h,jt	  j  ju	  j�  jv	  j  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j-  jv	  j  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j4  jv	  j  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j�  ju	  j>  jv	  j  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hK K�r�  h	hj0  h,jt	  j�  ju	  j  jv	  j  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  jH  jv	  jf  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j�  jv	  jf  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK
�r�  h	hj0  h,jt	  j  ju	  jP  jv	  jf  j3  Nubh)�r�  }r�  (hKhK K�r�  hKK�r�  h	hj0  h,jt	  j  ju	  jZ  jv	  jf  j3  Nubh)�r�  }r�  (hKhKK	�r�  hK K�r�  h	hj0  h,jt	  j  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK	�r�  hKK�r�  h	hj0  h,jt	  j  ju	  ju  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK	�r�  hKK�r�  h	hj0  h,jt	  j  ju	  jk  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK	�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hK K�r�  h	hj0  h,jt	  j	  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  ju  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  jk  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK	�r�  h	hj0  h,jt	  j	  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j,  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j4  jv	  j�  j3  Nubh)�r�  }r�  (hKhK	K
�r�  hKK�r�  h	hj0  h,jt	  j	  ju	  j@  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  j  ju	  jd  jv	  j�  j3  Nubh)�r�  }r   (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  ju  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  jk  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r	  hKK	�r
  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hK	K
�r  h	hj0  h,jt	  j  ju	  j	  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  j'  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  j,  jv	  j�  j3  Nubh)�r  }r  (hKhKK�r  hKK�r  h	hj0  h,jt	  j  ju	  j4  jv	  j�  j3  Nubh)�r  }r   (hKhKK�r!  hKK�r"  h	hj0  h,jt	  j  ju	  j@  jv	  j�  j3  Nubh)�r#  }r$  (hKhKK�r%  hK K�r&  h	hj0  h,jt	  j  ju	  jd  jv	  j�  j3  Nubh)�r'  }r(  (hKhKK�r)  hKK�r*  h	hj0  h,jt	  j  ju	  ju  jv	  j�  j3  Nubh)�r+  }r,  (hKhKK�r-  hKK�r.  h	hj0  h,jt	  j  ju	  jk  jv	  j�  j3  Nubh)�r/  }r0  (hKhKK�r1  hKK	�r2  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r3  }r4  (hKhKK�r5  hK	K
�r6  h	hj0  h,jt	  j  ju	  j	  jv	  j�  j3  Nubh)�r7  }r8  (hKhKK�r9  hKK�r:  h	hj0  h,jt	  j  ju	  j  jv	  j�  j3  Nubh)�r;  }r<  (hKhKK�r=  hKK�r>  h	hj0  h,jt	  j  ju	  j'  jv	  j�  j3  Nubh)�r?  }r@  (hKhKK�rA  hKK�rB  h	hj0  h,jt	  j  ju	  j,  jv	  j�  j3  Nubh)�rC  }rD  (hKhKK�rE  hKK�rF  h	hj0  h,jt	  j  ju	  j4  jv	  j�  j3  Nubh)�rG  }rH  (hKhKK�rI  hKK�rJ  h	hj0  h,jt	  j  ju	  j@  jv	  j�  j3  Nubh)�rK  }rL  (hKhKK�rM  hK K�rN  h	hj0  h,jt	  j'  ju	  jd  jv	  j�  j3  Nubh)�rO  }rP  (hKhKK�rQ  hKK�rR  h	hj0  h,jt	  j'  ju	  ju  jv	  j�  j3  Nubh)�rS  }rT  (hKhKK�rU  hKK�rV  h	hj0  h,jt	  j'  ju	  jk  jv	  j�  j3  Nubh)�rW  }rX  (hKhKK�rY  hKK	�rZ  h	hj0  h,jt	  j'  ju	  j  jv	  j�  j3  Nubh)�r[  }r\  (hKhKK�r]  hK K�r^  h	hj0  h,jt	  j,  ju	  jd  jv	  j�  j3  Nubh)�r_  }r`  (hKhKK�ra  hKK�rb  h	hj0  h,jt	  j,  ju	  ju  jv	  j�  j3  Nubh)�rc  }rd  (hKhKK�re  hKK�rf  h	hj0  h,jt	  j,  ju	  jk  jv	  j�  j3  Nubh)�rg  }rh  (hKhKK�ri  hKK	�rj  h	hj0  h,jt	  j,  ju	  j  jv	  j�  j3  Nubh)�rk  }rl  (hKhKK�rm  hK	K
�rn  h	hj0  h,jt	  j,  ju	  j	  jv	  j�  j3  Nubh)�ro  }rp  (hKhKK�rq  hKK�rr  h	hj0  h,jt	  j,  ju	  j  jv	  j�  j3  Nubh)�rs  }rt  (hKhKK�ru  hKK�rv  h	hj0  h,jt	  j,  ju	  j  jv	  j�  j3  Nubh)�rw  }rx  (hKhKK�ry  hKK�rz  h	hj0  h,jt	  j,  ju	  j'  jv	  j�  j3  Nubh)�r{  }r|  (hKhKK�r}  hKK�r~  h	hj0  h,jt	  j,  ju	  j4  jv	  j�  j3  Nubh)�r  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j,  ju	  j@  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  j4  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  ju  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  jk  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj0  h,jt	  j4  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK	K
�r�  h	hj0  h,jt	  j4  ju	  j	  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  j,  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j4  ju	  j@  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K�r�  h	hj0  h,jt	  j@  ju	  jd  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  ju  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  jk  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK	�r�  h	hj0  h,jt	  j@  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK	K
�r�  h	hj0  h,jt	  j@  ju	  j	  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  j  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  j'  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  j,  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j@  ju	  j4  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j  jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j   jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j  jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j  ju	  j   jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j   ju	  j  jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j   ju	  j  jv	  jk  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  jP  ju	  jJ  jv	  j�  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj0  h,jt	  jZ  ju	  j�  jv	  j"	  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj0  h,jt	  j`  ju	  j�  jv	  j"	  j3  Nubh)�r�  }r�  (hKhKK�r�  hK K �r�  h	hj0  h,jt	  j`  ju	  j�  jv	  j"	  j3  Nubh)�r�  }r�  (hKhKK�r�  hKK�r�  h	hj0  h,jt	  j`  ju	  jZ  jv	  j"	  j3  Nubh)�r�  }r   (hKhKK�r  hKK	�r  h	hj0  h,jt	  j`  ju	  jj  jv	  j"	  j3  Nubh)�r  }r  (hKhKK�r  hK
K�r  h	hj0  h,jt	  j`  ju	  jt  jv	  j"	  j3  Nubh)�r  }r  (hKhKK	�r	  hK K �r
  h	hj0  h,jt	  jj  ju	  j�  jv	  j"	  j3  Nubh)�r  }r  (hKhKK	�r  hK K �r  h	hj0  h,jt	  jj  ju	  j�  jv	  j"	  j3  Nubh)�r  }r  (hKhKK	�r  hKK�r  h	hj0  h,jt	  jj  ju	  jZ  jv	  j"	  j3  Nubh)�r  }r  (hKhKK	�r  hKK�r  h	hj0  h,jt	  jj  ju	  j`  jv	  j"	  j3  Nubh)�r  }r  (hKhKK	�r  hK
K�r  h	hj0  h,jt	  jj  ju	  jt  jv	  j"	  j3  Nubh)�r  }r  (hKhK
K�r  hK K �r  h	hj0  h,jt	  jt  ju	  j�  jv	  j"	  j3  Nubh)�r  }r   (hKhK
K�r!  hK K �r"  h	hj0  h,jt	  jt  ju	  j�  jv	  j"	  j3  Nubh)�r#  }r$  (hKhK
K�r%  hKK�r&  h	hj0  h,jt	  jt  ju	  jZ  jv	  j"	  j3  Nubh)�r'  }r(  (hKhK
K�r)  hKK�r*  h	hj0  h,jt	  jt  ju	  j`  jv	  j"	  j3  Nubh)�r+  }r,  (hKhK
K�r-  hKK	�r.  h	hj0  h,jt	  jt  ju	  jj  jv	  j"	  j3  Nube.