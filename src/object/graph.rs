//! Dependency-graph validation.
//!
//! Object graphs must be acyclic (unless a future representation explicitly
//! legalizes cycles) and depth-bounded. Validation is a bounded DFS over the
//! store; hostile input cannot produce pathological recursion because the
//! visit set and the depth cap (`MAX_REFERENCE_DEPTH`, `MAX_GRAPH_NODES`)
//! bound the walk.

use crate::error::{Error, Result};
use crate::object::id::ContentId;
use crate::object::{ObjectData, ObjectId, ObjectStore, SampleObject};
use std::collections::HashSet;

/// Direct content dependencies of an object (its edges).
fn outgoing(store: &ObjectStore, id: ObjectId) -> Vec<ContentId> {
    match store.get(id) {
        Ok(obj) => match &obj.data {
            ObjectData::Literal(_) => Vec::new(),
            ObjectData::Referenced(r) => vec![r.target_content],
        },
        Err(_) => Vec::new(),
    }
}

/// Depth-bounded acyclicity check from `root` over the whole reachable graph.
pub fn check_acyclic(store: &ObjectStore, root: ObjectId) -> Result<()> {
    let mut visiting: HashSet<ContentId> = HashSet::new();
    let mut done: HashSet<ContentId> = HashSet::new();
    let mut visited_count: u32 = 0;

    fn walk(
        store: &ObjectStore,
        id: ObjectId,
        visiting: &mut HashSet<ContentId>,
        done: &mut HashSet<ContentId>,
        visited_count: &mut u32,
        depth: u32,
    ) -> Result<()> {
        if depth > crate::limits::MAX_REFERENCE_DEPTH {
            return Err(Error::dependency(format!(
                "graph depth exceeds {} at {id}",
                crate::limits::MAX_REFERENCE_DEPTH
            )));
        }
        let obj: &SampleObject = store.get(id)?;
        *visited_count += 1;
        if *visited_count > crate::limits::MAX_GRAPH_NODES {
            return Err(Error::limit("graph node budget exceeded"));
        }
        let cid = obj.content_id;
        if done.contains(&cid) {
            return Ok(());
        }
        if !visiting.insert(cid) {
            return Err(Error::dependency(format!(
                "dependency cycle detected at {id} (content {cid})"
            )));
        }
        for dep in outgoing(store, id) {
            let next = store
                .id_of_content(&dep)
                .ok_or_else(|| Error::dependency(format!("{id} -> missing content {dep}")))?;
            walk(store, next, visiting, done, visited_count, depth + 1)?;
        }
        visiting.remove(&cid);
        done.insert(cid);
        Ok(())
    }

    walk(store, root, &mut visiting, &mut done, &mut visited_count, 0)
}

/// Validate the entire store.
pub fn validate_store(store: &ObjectStore) -> Result<()> {
    for obj in store.iter() {
        check_acyclic(store, obj.id)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Literal;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::object::reference::Referenced;
    use crate::universe::layout::Layout;

    fn ref_obj(target: ContentId) -> (ObjectDescriptor, ObjectData) {
        let d = ObjectDescriptor::new(Representation::Referenced, 4, Layout::Mono, None).unwrap();
        let r = Referenced::checked(target, 1 << 24, None).unwrap();
        (d, ObjectData::Referenced(r))
    }

    #[test]
    fn chain_of_references_is_acyclic() {
        let mut store = ObjectStore::new();
        let ld = ObjectDescriptor::new(Representation::Literal, 4, Layout::Mono, None).unwrap();
        let root = store
            .insert(
                ld.clone(),
                ObjectData::Literal(Literal::new(&ld, vec![0; 4]).unwrap()),
            )
            .unwrap();
        let root_content = store.get(root).unwrap().content_id;
        // Chain of 3 references to the literal.
        let (d1, o1) = ref_obj(root_content);
        let c1 = crate::object::canonical_content_id(&d1, &o1).unwrap();
        let _ = store.insert(d1, o1).unwrap();
        let (d2, o2) = ref_obj(c1);
        let c2 = crate::object::canonical_content_id(&d2, &o2).unwrap();
        let _ = store.insert(d2, o2).unwrap();
        let (d3, o3) = ref_obj(c2);
        let id3 = store.insert(d3, o3).unwrap();
        check_acyclic(&store, id3).expect("acyclic");
        validate_store(&store).expect("store valid");
    }

    #[test]
    fn depth_bound_triggers_on_long_chain() {
        let mut store = ObjectStore::new();
        let ld = ObjectDescriptor::new(Representation::Literal, 1, Layout::Mono, None).unwrap();
        let root = store
            .insert(
                ld.clone(),
                ObjectData::Literal(Literal::new(&ld, vec![0]).unwrap()),
            )
            .unwrap();
        let mut target = store.get(root).unwrap().content_id;
        // Build a chain longer than MAX_REFERENCE_DEPTH (64) + 1.
        let n = crate::limits::MAX_REFERENCE_DEPTH + 2;
        for _ in 0..n {
            let (d, o) = ref_obj(target);
            let content = crate::object::canonical_content_id(&d, &o).unwrap();
            let id = store.insert(d, o).unwrap();
            target = store.get(id).unwrap().content_id;
            // canonical_content_id must equal stored content id
            assert_eq!(target, content);
        }
        let last = store.id_of_content(&target).unwrap();
        assert!(check_acyclic(&store, last).is_err());
    }
}
