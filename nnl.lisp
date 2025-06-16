(asdf:defsystem :nnl
  :license "MIT"
  :version "0.1.0"
  :homepage "https://github.com/danish-song-of-liberation/nnl"
  :description "Common lisp neural networks mvp framework"
  :depends-on (:fiveam :magicl)
  :serial t
  :components ((:module "src"
				:components ((:module "math"
							  :components ((:module "autodiff"
											:components ((:file "autodiff" :type "lisp")
														 (:file "autodiff-d" :type "lisp")
														 (:file "autodiff-h" :type "lisp")))
											(:file "numgrad" :type "lisp")))))))
											