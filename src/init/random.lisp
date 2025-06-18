(in-package :nnl.init)

(defun random-initialize (shapes &key (from -1.0) (to 1.0) (requires-grad nil))
  (nnl.math:randn shapes :from from :to to :requires-grad requires-grad))
  