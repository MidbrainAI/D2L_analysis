from serving_cost import GEMMA2_2B, QWEN3_4B, account, kv_bytes, reader_macs_per_token, request_cost


def test_gemma_kv_bytes_match_the_architecture_and_window():
    assert kv_bytes(GEMMA2_2B, 1) == 106_496
    assert kv_bytes(GEMMA2_2B, 150) == 150 * 106_496
    # Above the 4,096 window the local half of the layers stops growing.
    assert kv_bytes(GEMMA2_2B, 8192) < 8192 * 106_496
    assert kv_bytes(QWEN3_4B, 10) == 10 * 36 * 2 * 8 * 128 * 2


def test_d2l_cost_does_not_grow_with_memory_length():
    short = request_cost(GEMMA2_2B, "d2l", memory_tokens=150, documents=1, question_tokens=40, output_tokens=32)
    long = request_cost(GEMMA2_2B, "d2l", memory_tokens=4000, documents=1, question_tokens=40, output_tokens=32)
    assert short["request_macs"] == long["request_macs"] and short["prefill_positions"] == 40
    icl = request_cost(GEMMA2_2B, "icl", memory_tokens=150, documents=1, question_tokens=40, output_tokens=32)
    cached = request_cost(GEMMA2_2B, "icl_prefix_cache", memory_tokens=150, documents=1, question_tokens=40,
                          output_tokens=32)
    assert icl["prefill_positions"] == 190 and cached["prefill_positions"] == 40
    assert icl["decode_macs"] == cached["decode_macs"] and cached["request_macs"] < icl["request_macs"]
    assert short["per_user_resident_bytes"] == 26 * 16 * (9216 + 2304) * 2


def test_factored_reader_is_cheaper_and_grows_with_slots():
    assert reader_macs_per_token(GEMMA2_2B, 128, True) < reader_macs_per_token(GEMMA2_2B, 128, False)
    assert reader_macs_per_token(GEMMA2_2B, 128, False) > reader_macs_per_token(GEMMA2_2B, 8, False)


def test_account_reports_break_even_queries():
    report = account(GEMMA2_2B, lengths=[150, 2048], documents=[1, 4], question_tokens=40, output_tokens=32,
                     document_tokens=150)
    long = report["by_memory_length"][1]["break_even_queries"]
    assert long["d2l_vs_icl"] is not None and long["d2l_vs_icl"] > 0
    assert set(report["by_document_count"][1]["break_even_queries"]) == {
        f"{a}_vs_{b}" for a in ("d2l", "d2l_reader", "d2l_reader_factored") for b in ("icl", "icl_prefix_cache")}
