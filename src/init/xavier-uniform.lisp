(in-package :nnl.init)

(defun xavier-uniform-initialize (shapes sum-in-out &key (requires-grad nil))
  (let* ((ceil (sqrt (/ 6 sum-in-out)))
		 (floor (- ceil)))
		 
	(random-initialize shapes :from floor :to ceil :requires-grad requires-grad)))
	