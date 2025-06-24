(in-package :nnl.hli)

(defmacro sequential (&body layers)
  `(nnl.nn:sequential ,@layers))

(defmacro fc (i arrow o &body keys)
  `(nnl.nn:fc :i ,i :o ,o ,@keys))

(defmacro mlp (neurons &body keys)
  `(nnl.nn:mlp :order (loop for item in ,neurons when (identity item) collect item) ,@keys))
  