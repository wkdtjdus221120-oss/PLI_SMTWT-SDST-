from aco_smtwt_sdst import solve_aco
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Ant Colony Optimization on SMTWT-SDST.")
    parser.add_argument("--ants", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--evaporation-rate", type=float, default=0.20)
    parser.add_argument("--elitist-weight", type=float, default=2.0)
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_aco(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        n_ants=args.ants,
        alpha=args.alpha,
        beta=args.beta,
        evaporation_rate=args.evaporation_rate,
        elitist_weight=args.elitist_weight,
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
