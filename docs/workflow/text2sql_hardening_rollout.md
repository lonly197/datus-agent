# Text2SQL & SQL Review Hardening - Rollout Plan

## Overview

This rollout plan implements the three-phase hardening of Text2SQL and SQL Review workflows to reduce invalid SQL outputs, provide real-time actionable errors to frontends, and improve robustness when table/schema resources are missing.

## 🎯 Goals

- **Reduce invalid Text2SQL outputs** by 80% through early syntax validation
- **Provide real-time error feedback** to frontends with actionable suggestions
- **Improve robustness** when database resources are unavailable
- **Maintain backward compatibility** through feature flags

## 📋 Implementation Phases

### Phase 1: SQL Syntax Precheck (✅ Completed)
**Risk Level**: Low
**Duration**: 2-3 days

#### Changes Made
- Added `validate_sql_syntax()` method to `DBFuncTool` class
- Integrated syntax validation into `Text2SQLExecutionMode`
- Added comprehensive unit tests for SQL syntax validation

#### Configuration
```yaml
plan_executor:
  text2sql_preflight:
    enabled: false  # Default: disabled for backward compatibility
    syntax_validation: true
    continue_on_failure: true  # Non-blocking during rollout
```

#### Acceptance Criteria
- `validate_sql_syntax` returns expected structure for valid/invalid SQL
- Invalid SQL triggers SSE ErrorEvent with clear message
- Feature controlled by `text2sql_preflight.syntax_validation` flag

### Phase 2: Enhanced Error/Event Feedback (✅ Completed)
**Risk Level**: Medium
**Duration**: 3-4 days

#### Changes Made
- Added error event helper methods in `ChatAgenticNode`:
  - `_send_syntax_error_event()`
  - `_send_table_not_found_event()`
  - `_send_db_connection_error_event()`
- Extended error classification in `plan_hooks.py`
- Integrated error emission into preflight and server executor paths
- Added unit tests for SSE ErrorEvent payloads

#### Configuration
```yaml
plan_executor:
  text2sql_preflight:
    error_events:
      send_syntax_errors: true
      send_table_errors: false  # Usually not needed for Text2SQL
      send_connection_errors: true
```

#### Acceptance Criteria
- Syntax, table-not-found, and connection errors emit SSE ErrorEvents
- Error events include `errorType`, `suggestions`, and `canRetry` fields
- Frontend-visible messages are concise and actionable

### Phase 3: Intelligent Table Handling & Fallbacks (✅ Completed)
**Risk Level**: Medium-High
**Duration**: 4-6 days

#### Changes Made
- Implemented `check_table_exists()` DB tool with caching
- Added dynamic tool-sequence adjustment logic
- Implemented search_table fallback when describe_table fails
- Added comprehensive unit tests for table existence handling

#### Configuration
```yaml
plan_executor:
  text2sql_preflight:
    table_existence_check:
      enabled: true
      on_missing: "continue"  # Text2SQL continues even if tables missing
      max_suggestions: 3
```

#### Acceptance Criteria
- Tool sequence adjusts dynamically based on table existence
- Missing table triggers search_table fallback with suggestions
- SSE includes suggestions when fallback finds candidates
- Caching reduces repeated calls for table existence checks

## 🚀 Rollout Strategy

### Phase 1 Rollout (Week 1)
**Target**: Enable syntax validation in non-blocking mode

1. **Preparation (Day 1)**
   - Deploy code changes to staging environment
   - Run unit tests: `pytest tests/unit_tests/test_db_func_tools.py::TestSQLSyntaxValidation`
   - Verify no regressions in existing Text2SQL functionality

2. **Gradual Rollout (Day 2-3)**
   - Enable syntax validation with `continue_on_failure: true`
   ```yaml
   plan_executor:
     text2sql_preflight:
       enabled: true
       syntax_validation: true
       continue_on_failure: true  # Non-blocking
   ```
   - Monitor error logs for syntax validation issues
   - Track SSE ErrorEvent emission

3. **Validation (Day 3)**
   - Verify syntax errors are caught and reported
   - Confirm no impact on successful queries
   - Check frontend error handling

### Phase 2 Rollout (Week 2)
**Target**: Enable error event emission

1. **Enable Error Events**
   ```yaml
   plan_executor:
     text2sql_preflight:
       error_events:
         send_syntax_errors: true
         send_connection_errors: true
   ```

2. **Frontend Integration**
   - Verify frontend handles new ErrorEvent types
   - Test error message display
   - Validate suggestion presentation

### Phase 3 Rollout (Week 3-4)
**Target**: Enable intelligent table handling

1. **Enable Table Checks**
   ```yaml
   plan_executor:
     text2sql_preflight:
       table_existence_check:
         enabled: true
         on_missing: "continue"
   ```

2. **Fallback Testing**
   - Test scenarios with missing tables
   - Verify search_table fallback works
   - Check suggestion quality and relevance

## 📊 Monitoring & Metrics

### Key Metrics to Track

1. **Syntax Validation Success Rate**
   ```
   SELECT
     COUNT(*) as total_validations,
     SUM(CASE WHEN syntax_valid THEN 1 ELSE 0 END) as successful_validations,
     ROUND(100.0 * SUM(CASE WHEN syntax_valid THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
   FROM text2sql_syntax_validations
   WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
   ```

2. **Error Event Emission**
   ```
   SELECT
     error_type,
     COUNT(*) as event_count,
     AVG(can_retry) as avg_retryable_rate
   FROM error_events
   WHERE workflow_type = 'text2sql'
     AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
   GROUP BY error_type
   ```

3. **Table Existence Check Performance**
   ```
   SELECT
     AVG(cache_hit_rate) as avg_cache_hit_rate,
     AVG(execution_time_ms) as avg_execution_time,
     COUNT(*) as total_checks
   FROM table_existence_checks
   WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
   ```

### Monitoring Dashboard

```sql
-- Text2SQL Health Dashboard Query
SELECT
  'syntax_validation' as metric,
  ROUND(100.0 * successful_validations / total_validations, 2) as value,
  'percentage' as unit
FROM (
  SELECT
    COUNT(*) as total_validations,
    SUM(CASE WHEN syntax_valid THEN 1 ELSE 0 END) as successful_validations
  FROM text2sql_syntax_validations
  WHERE timestamp >= CURRENT_DATE - INTERVAL '1 hour'
) v

UNION ALL

SELECT
  'error_events_per_minute' as metric,
  COUNT(*) / 60.0 as value,
  'events/min' as unit
FROM error_events
WHERE workflow_type = 'text2sql'
  AND timestamp >= CURRENT_DATE - INTERVAL '1 hour'

UNION ALL

SELECT
  'table_check_cache_hit_rate' as metric,
  ROUND(100.0 * AVG(cache_hit_rate), 2) as value,
  'percentage' as unit
FROM table_existence_checks
WHERE timestamp >= CURRENT_DATE - INTERVAL '1 hour'
```

## 🔄 Rollback Plan

### Immediate Rollback (Feature Flags)
All features can be disabled instantly via configuration:

```yaml
plan_executor:
  text2sql_preflight:
    enabled: false  # Disables all new functionality
```

### Gradual Rollback (Per Feature)
```yaml
plan_executor:
  text2sql_preflight:
    syntax_validation: false
    table_existence_check:
      enabled: false
    error_events:
      send_syntax_errors: false
      send_connection_errors: false
```

### Code Rollback
- Keep feature flags in place for future re-enablement
- Rollback commits in reverse order if needed
- Maintain database schema compatibility

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] Text2SQL queries with syntax errors are caught before execution
- [ ] Frontend receives ErrorEvents with actionable suggestions
- [ ] Missing tables trigger appropriate fallbacks and suggestions
- [ ] All existing Text2SQL functionality remains unchanged
- [ ] SQL Review functionality is not impacted

### Performance Requirements
- [ ] Syntax validation adds <100ms latency per query
- [ ] Table existence checks are cached effectively (>90% hit rate)
- [ ] Error event emission doesn't impact normal query throughput
- [ ] Memory usage remains within acceptable bounds

### Quality Requirements
- [ ] Unit test coverage >95% for new functionality
- [ ] Integration tests pass in staging environment
- [ ] No regressions in existing test suites
- [ ] Documentation updated and reviewed

## 📈 Success Metrics

### Primary Metrics (Week 1-2)
- Syntax validation catch rate: >90% of obvious syntax errors
- Error event emission: 100% for caught errors
- Frontend error handling: No crashes or display issues

### Secondary Metrics (Week 3-4)
- Table existence check cache hit rate: >85%
- Fallback suggestion relevance: >70% user satisfaction
- Overall Text2SQL success rate improvement: +15%

### Long-term Metrics (Month 2+)
- Reduction in invalid SQL generation: >50%
- User-reported error improvement: >60%
- System reliability improvement: >30%

## 📋 Post-Rollout Activities

1. **Monitor and Tune** (Ongoing)
   - Adjust cache TTL based on usage patterns
   - Refine error message suggestions based on user feedback
   - Optimize performance bottlenecks

2. **User Training** (Week 5)
   - Update frontend error handling for new event types
   - Document new error messages and suggestions
   - Train support team on new error scenarios

3. **Feature Expansion** (Month 2)
   - Consider enabling features for SQL Review workflows
   - Add more sophisticated fallback strategies
   - Implement learning from user corrections

## 🔗 Dependencies

### Required Before Rollout
- [ ] Frontend ErrorEvent handling implementation
- [ ] Monitoring dashboard setup
- [ ] Staging environment validation
- [ ] Rollback procedures documented

### Required During Rollout
- [ ] 24/7 monitoring team availability
- [ ] Frontend team coordination
- [ ] Database performance monitoring

## 📞 Support Plan

### During Rollout
- **Primary Contact**: Development team lead
- **Secondary Contact**: DevOps/SRE team
- **Escalation Path**: Product manager → Engineering manager → CTO

### Monitoring Windows
- **Phase 1**: Business hours monitoring (9 AM - 6 PM local time)
- **Phase 2**: Extended monitoring (8 AM - 8 PM local time)
- **Phase 3**: 24/7 monitoring during first 48 hours

### Emergency Contacts
- **Code Issues**: Development team Slack channel
- **Infrastructure Issues**: DevOps/SRE on-call
- **User Impact**: Product/support team

---

**Rollout Owner**: Development Team
**Approval Required**: Engineering Manager, Product Manager
**Go-Live Date**: [To be determined]
**Rollback Window**: 1 week post-go-live
