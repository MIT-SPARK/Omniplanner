(define (domain goto-object-domain)
    (:requirements :derived-predicates :typing :adl)
    (:types
        point-of-interest - object
        place dsg_object - point-of-interest
    )


    (:predicates
        (at-poi ?p - point-of-interest)
        (connected ?s - point-of-interest ?t - point-of-interest)
        (suspicious ?o - dsg_object)
        (at-object ?o)
        (at-place ?p)

        (visited-poi ?p)
        (visited-place ?p)
        (visited-object ?o)

        (safe ?o)
    )

    (:functions
        (distance ?s ?t)
        (total-cost)
    )

    (:derived (at-object ?o - dsg_object)
        (at-poi ?o))

    (:derived (at-place ?p - place)
        (at-poi ?p))

    (:derived (visited-place ?p - place)
        (visited-poi ?p))

    (:derived (visited-object ?p - dsg_object)
        (visited-poi ?p))

    (:derived (safe ?o - dsg_object)
        (not (suspicious ?o)))


    (:action goto-poi
        :parameters (?s - point-of-interest ?t - point-of-interest)
        :precondition (and (at-poi ?s) (or (connected ?s ?t)
                                           (connected ?t ?s)))
        :effect (and (not (at-poi ?s))
                     (at-poi ?t)
                     (visited-poi ?t)
                     (increase (total-cost) (distance ?s ?t))
        )
    )

    (:action inspect
     :parameters (?o - dsg_object)
     :precondition (at-object ?o)
     :effect (and (not (suspicious ?o))
                  (increase (total-cost) 1)
            )
     )

)
