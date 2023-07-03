from rdds.lib.resource_usage import ProcessResourceUsage


def test_resource_usage():
    """
    Test ProcessResourceUsage API.

    It's difficult to reliably write a test that expects fixed values
    at runtime. Because of this just consider the API and runtime issues,
    and not the return values.
    """
    # GIVEN an instance of ProcessResourceUsage
    stats = ProcessResourceUsage()
    # WHEN reading the runtime statistics
    # THEN expect it to succeed
    _ = '%s' % str(stats)
