from ts_smtwt_sdst import solve_ts
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Tabu Search on SMTWT-SDST.")
    parser.add_argument("--tabu-tenure", type=int, default=10)
    parser.add_argument(
        "--neighborhood-sample",
        type=int,
        default=300,
        help="Swap moves sampled per iteration; 0 means full neighborhood.",
    )
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_ts(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        tabu_tenure=args.tabu_tenure,
        neighborhood_sample=(
            None if args.neighborhood_sample == 0 else args.neighborhood_sample
        ),
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
