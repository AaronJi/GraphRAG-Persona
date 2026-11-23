"""Run GraphRAG skill retrieval and user-personalized dialogue generation."""

from scripts.pipeline import (
    initialize_embedder,
    initialize_llm,
    initialize_vl,
    load_samples,
    parse_args,
    prepare_index,
    run_generation,
    run_profile_inference,
    run_retrieval,
    save_sample_results,
)


def main():
    args = parse_args()
    embedder = initialize_embedder(args)
    skill_index = prepare_index(embedder, args)
    vl_model = initialize_vl(args)
    llm = initialize_llm(args)
    samples = load_samples(args)

    for sample in samples:
        profile, vl_seconds = run_profile_inference(vl_model, sample, args, llm)
        retrieval = run_retrieval(embedder, skill_index, profile, args)
        generation = run_generation(llm, sample, profile, retrieval, args)
        save_sample_results(sample, profile, vl_seconds, retrieval, generation, args)


if __name__ == "__main__":
    main()
