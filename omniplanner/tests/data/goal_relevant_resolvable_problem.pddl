(define (problem object-rearrangement-domain)
        (:domain object-rearrangement-domain)
        (:objects
pstart p0 - place
o0 - dsg_object
)
        (:init
 (= (total-cost) 0)
(at-poi pstart)
(connected pstart o0)
(= (distance pstart o0) 1)
(= (distance o0 pstart) 1)
(connected pstart p0)
(= (distance pstart p0) 1)
(= (distance p0 pstart) 1)
(connected o0 p0)
(= (distance o0 p0) 0)
(= (distance p0 o0) 0)
(object-in-place o0 p0) )
        (:goal (and (visited-object o0)))
        
(:metric minimize (total-cost)))