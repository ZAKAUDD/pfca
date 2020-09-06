# Programming Framework for Clinical Image Analysis
<p>PFCA(Programming Framework for Clinical Image Analysis(PFCA) is an open-source python library to preprocess and analyse radiological images, mainly MRI. The repository is the result of my efforts in the undergraduate thesis that mainly focused on the detection of cerebral microbleeds in MRI images of patients. The library contains helper functions for frequently used preprocessing operations in MRI. I have also curated the 3d visualisation functions in Mayavi to visualise the brain segments.</p>
<p><b>Main Contributors:</b>
<ul>
  <li>Shwetank Panwar, IIT Guwahati</li>
</ul>

</p>  

<p><b>File Description:</b></p>
<ul>
  <li><b>core</b> : The core submodule contains the main preprocessing and postprocessing operations required to work with raw DICOM data obtained from clinics or radiologists.</li>
  <li><b>exp</b> : The experiment submodule consists of the functions required while trying to analyse the initial patterns in data using visualization tools like PCA, tSNE and Isomap.</li>
  <li><b>file_read</b> : The submodule contains important helper functions to correctly preprocess and read raw DICOM files.</li>
  <li><b>visuals</b> : The visualization submodule contains all the interesting functions to develop beutiful visualisations from the MRI datasets. In the thesis work, it was mainly used to visualise the size of microbleeds as compared to the size of patient's brain.</li>
</ul>

