(in-package :nnl.init)

(defun xavier-normal-initialize (shapes sum-in-out &key (requires-grad nil))
  (let* ((ceil (sqrt (/ 2 sum-in-out)))
		 (floor (- ceil)))
		 
	(random-initialize shapes :from floor :to ceil :requires-grad requires-grad)))
	