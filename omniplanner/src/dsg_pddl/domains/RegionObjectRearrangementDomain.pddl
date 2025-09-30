(define (domain region-object-rearrangement-domain)
    (:requirements :derived-predicates :typing :adl)
    (:types
        region point-of-interest - object
        place dsg_object - point-of-interest
    )


    (:predicates
        (at-poi ?p - point-of-interest)
        (connected ?s - point-of-interest ?t - point-of-interest)
        (suspicious ?o - dsg_object)
        (at-object ?o)
        (at-place ?p)
        (in-region ?r)

        (holding ?o - dsg_object)
        (hand-full)
        (object-in-place ?o - dsg_object ?p - place)
        (place-in-region ?p - place ?r - region)

        (visited-poi ?p)
        (visited-place ?p)
        (visited-object ?o)
        (visited-region ?r)

        (safe ?o)
        (explored-region ?r - region)
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

    (:derived (in-region ?r - region)
        (exists (?p - place) (and (at-place ?p) (place-in-region ?p ?r))))

    (:derived (visited-region ?r - region)
        (exists (?p - place) (and (visited-place ?p) (place-in-region ?p ?r))))

    (:derived (explored-region ?r - region)
        (forall (?p - place)
            (implies (place-in-region ?p ?r)
                     (visited-place ?p))))


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

    (:action pick-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (not (hand-full))
                        (safe ?o)
                        (at-object ?o)
                        (object-in-place ?o ?p))
     :effect (and (holding ?o)
                  (hand-full)
                  (not (object-in-place ?o ?p))))

    (:action place-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (holding ?o) (at-place ?p))
     :effect (and (not (holding ?o))
                  (not (hand-full))
                  (object-in-place ?o ?p)))

)
