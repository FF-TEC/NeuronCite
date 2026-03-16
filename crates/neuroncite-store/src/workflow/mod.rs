// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Workflow layer: job tracking and idempotency for the indexing pipeline.
//
// - `job` -- Tracks the state of indexing jobs (queued, running, completed,
//   failed, canceled) and their progress (items processed / total items).
//   A periodic cleanup task deletes completed/failed jobs whose finished_at
//   timestamp is older than 24 hours.
// - `idempotency` -- Manages idempotency keys that prevent duplicate processing
//   when an indexing operation is interrupted and resumed. Each idempotency entry
//   links a client-supplied key to a job and session.

pub mod idempotency;
pub mod job;
