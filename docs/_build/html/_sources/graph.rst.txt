Graph of Modules and Functions
==============================

.. graphviz::

   digraph G {
       rankdir=LR;
       node [shape=box];

       dsp [label="Digital Signal Processing"];
       detect [label="Detection"];
       data_handle [label="DAS data handling"];
       get_fx [label="dsp.get_fx"];
       gen_linear_chirp [label="detect.gen_linear_chirp"];

       dsp -> get_fx;
       detect -> gen_linear_chirp;