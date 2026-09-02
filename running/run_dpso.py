from dpso_smtwt_sdst import solve_dpso
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Discrete PSO on SMTWT-SDST.")
    parser.add_argument("--swarm-size", type=int, default=40)
    parser.add_argument("--inertia-probability", type=float, default=0.50)
    parser.add_argument("--cognitive-probability", type=float, default=0.60)
    parser.add_argument("--social-probability", type=float, default=0.70)
    parser.add_argument("--mutation-probability", type=float, default=0.05)
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_dpso(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        swarm_size=args.swarm_size,
        inertia_probability=args.inertia_probability,
        cognitive_probability=args.cognitive_probability,
        social_probability=args.social_probability,
        mutation_probability=args.mutation_probability,
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
