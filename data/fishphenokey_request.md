# FishPhenoKey dataset access — draft request email

**To:** liuweizhen@whut.edu.cn
**Subject:** Request for FishPhenoKey dataset access — DeepLabCut pretraining for brook trout morphometrics

---

Dear Dr. Liu and FishPhenoKey authors,

I am a researcher at the Cornell University Museum of Vertebrates working on
automated linear and area morphometrics of preserved brook trout
(*Salvelinus fontinalis*) specimens. Our current workflow uses ImageJ for
manual landmark placement on lateral photographs, and we are migrating to a
DeepLabCut-based keypoint detection pipeline.

I came across your FishPhenoKey work and would like to ask whether the
dataset is available for academic / non-commercial research use. We would
like to use it as a **pretraining source** before fine-tuning DeepLabCut on
our own brook trout labels. Even if *Salvelinus fontinalis* is not in your
species coverage, the shared morphological structure across teleosts
(particularly the fin landmarks and body outline) should transfer well and
help compensate for the relatively small number of labeled images we can
produce in-house.

A few specific questions, if you are open to sharing:

1. **Access:** Is there a download link, DOI, or request form we should go
   through? Is a data-use agreement required?
2. **Species coverage:** Does the dataset include any salmonids
   (*Salmo*, *Oncorhynchus*, *Salvelinus*), or is it primarily other teleost
   families?
3. **Annotation format:** Are the keypoints released as COCO-style JSON,
   DeepLabCut CSV/H5, or another format? We would like to confirm
   compatibility before integrating.
4. **License / citation:** We are happy to cite the FishPhenoKey paper in
   any publication and acknowledge the dataset in our release notes.

Our end product will be an open-source morphometrics pipeline (currently at
an internal Cornell repository, with a planned public release) that reports
14 linear and 5 area measurements per specimen. Any models trained against
your data would be used strictly for research and would be documented as
such.

Thank you very much for considering this request, and for publishing
FishPhenoKey — it is a valuable contribution to fish phenomics.

Best regards,

**[Your name]**
Cornell University Museum of Vertebrates
[Your email] · [Institutional website / ORCID if applicable]

---

## Notes before sending

- Fill in your name, email, and any institutional signature block.
- If you have a Cornell advisor who should be CC'd (or who should send
  this instead, for weight), add them.
- If our public repo is ready at the time of sending, include the URL in
  the final paragraph so they can see exactly what the data would be used
  for.
- Consider attaching or linking: (a) the landmark schema
  (`src/fish_morpho/landmark_config.py`), (b) a sample of 2–3 preserved
  brook trout photos from the iDigBio harvest to make the "preserved
  specimen / lateral view" protocol concrete.
- Response time from Chinese academic addresses can be slow. If no reply
  in ~2 weeks, a polite follow-up is appropriate.
