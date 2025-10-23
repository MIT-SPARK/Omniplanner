(define (domain region-object-rearrangement-domain-multirobot-fd)
    (:requirements :derived-predicates :typing :adl)
    (:types
        region point-of-interest - object
        place dsg_object - point-of-interest
        robot - object
    )

    (:predicates
        (at-poi ?r - robot ?p - point-of-interest)
        (connected ?s - point-of-interest ?t - point-of-interest)
        (suspicious ?o - dsg_object)
        (at-object ?r - robot ?o - dsg_object)
        (at-place ?r - robot ?p - place)
        (in-region ?r - robot ?reg - region)

        (holding ?r - robot ?o - dsg_object)
        (hand-full ?r - robot)
        (object-in-place ?o - dsg_object ?p - place)
        (place-in-region ?p - place ?reg - region)

        (visited-poi ?p - point-of-interest)
        (visited-place ?p - place)
        (visited-object ?o - dsg_object)
        (visited-region ?reg - region)

        (safe ?o - dsg_object)
        (explored-region ?reg - region)
    )

    (:functions
        (distance ?s - point-of-interest ?t - point-of-interest)
        (total-cost)
    )

    (:derived (at-object ?r - robot ?o - dsg_object)
        (at-poi ?r ?o))

    (:derived (at-place ?r - robot ?p - place)
        (at-poi ?r ?p))

    (:derived (visited-place ?p - place)
        (visited-poi ?p))

    (:derived (visited-object ?o - dsg_object)
        (visited-poi ?o))

    (:derived (safe ?o - dsg_object)
        (not (suspicious ?o)))

    (:derived (in-region ?r - robot ?reg - region)
        (exists (?p - place) (and (at-place ?r ?p) (place-in-region ?p ?reg))))

    (:derived (visited-region ?reg - region)
        (exists (?p - place) (and (visited-place ?p) (place-in-region ?p ?reg))))

    (:derived (explored-region ?reg - region)
        (forall (?p - place)
            (or (not (place-in-region ?p ?reg))
                (visited-place ?p))))

    (:action goto-poi
        :parameters (?r - robot ?s - point-of-interest ?t - point-of-interest)
        :precondition (and (at-poi ?r ?s) (or (connected ?s ?t)
                                           (connected ?t ?s)))
        :effect (and (not (at-poi ?r ?s))
                     (at-poi ?r ?t)
                     (visited-poi ?t)
                     (increase (total-cost) (distance ?s ?t))
        )
    )

    (:action inspect
     :parameters (?r - robot ?o - dsg_object)
     :precondition (at-object ?r ?o)
     :effect (and (not (suspicious ?o))
                  (increase (total-cost) 1)
            )
     )

    (:action pick-object
     :parameters (?r - robot ?o - dsg_object ?p - place)
     :precondition (and (not (hand-full ?r))
                        (safe ?o)
                        (at-object ?r ?o)
                        (object-in-place ?o ?p))
     :effect (and (holding ?r ?o)
                  (hand-full ?r)
                  (not (object-in-place ?o ?p))))

    (:action place-object
     :parameters (?r - robot ?o - dsg_object ?p - place)
     :precondition (and (holding ?r ?o) (at-place ?r ?p))
     :effect (and (not (holding ?r ?o))
                  (not (hand-full ?r))
                  (object-in-place ?o ?p)))

)
