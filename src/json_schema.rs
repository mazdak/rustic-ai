use std::collections::HashMap;

use serde_json::{Map, Value};

const STRICT_INCOMPATIBLE_KEYS: [&str; 12] = [
    "minLength",
    "maxLength",
    "patternProperties",
    "unevaluatedProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "uniqueItems",
];

const STRICT_COMPATIBLE_STRING_FORMATS: [&str; 9] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
];

pub(crate) fn transform_openai_schema(schema: &Value, strict: Option<bool>) -> (Value, bool) {
    let mut transformer = OpenAIJsonSchemaTransformer::new(schema.clone(), strict);
    let mut schema = transformer.walk();
    if strict == Some(true) {
        enforce_strict_root(&mut schema);
    }
    (schema, transformer.is_strict_compatible)
}

fn enforce_strict_root(schema: &mut Value) {
    let Value::Object(map) = schema else {
        return;
    };

    let has_props = matches!(map.get("properties"), Some(Value::Object(_)));
    let is_object = map
        .get("type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value == "object")
        || has_props;

    if !is_object {
        return;
    }

    map.entry("type".to_string())
        .or_insert_with(|| Value::String("object".to_string()));
    map.insert("additionalProperties".to_string(), Value::Bool(false));

    if !map.contains_key("required")
        && let Some(Value::Object(properties)) = map.get("properties")
    {
        let required = properties
            .keys()
            .cloned()
            .map(Value::String)
            .collect::<Vec<_>>();
        map.insert("required".to_string(), Value::Array(required));
    }
}

struct OpenAIJsonSchemaTransformer {
    schema: Value,
    strict: Option<bool>,
    root_ref: Option<String>,
    defs: HashMap<String, Value>,
    legacy_defs: HashMap<String, Value>,
    is_strict_compatible: bool,
}

impl OpenAIJsonSchemaTransformer {
    fn new(schema: Value, strict: Option<bool>) -> Self {
        let root_ref = schema
            .get("$ref")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let defs = match schema.get("$defs") {
            Some(Value::Object(map)) => map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            _ => HashMap::new(),
        };
        let legacy_defs = match schema.get("definitions") {
            Some(Value::Object(map)) => map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            _ => HashMap::new(),
        };
        Self {
            schema,
            strict,
            root_ref,
            defs,
            legacy_defs,
            is_strict_compatible: true,
        }
    }

    fn walk(&mut self) -> Value {
        let mut root = match self.schema.clone() {
            Value::Object(map) => map,
            other => return other,
        };

        let defs = root.remove("$defs");
        let legacy_defs = root.remove("definitions");
        let handled = self.handle(Value::Object(root));

        let mut handled = match handled {
            Value::Object(map) => map,
            other => return other,
        };

        let mut handled_defs: Option<Map<String, Value>> = None;
        if let Some(Value::Object(defs_map)) = defs {
            let mut new_defs = Map::new();
            for (key, value) in defs_map {
                new_defs.insert(key, self.handle(value));
            }
            if !new_defs.is_empty() {
                handled_defs = Some(new_defs.clone());
                handled.insert("$defs".to_string(), Value::Object(new_defs));
            }
        }
        let mut handled_legacy_defs: Option<Map<String, Value>> = None;
        if let Some(Value::Object(defs_map)) = legacy_defs {
            let mut new_defs = Map::new();
            for (key, value) in defs_map {
                new_defs.insert(key, self.handle(value));
            }
            if !new_defs.is_empty() {
                handled_legacy_defs = Some(new_defs.clone());
                handled.insert("definitions".to_string(), Value::Object(new_defs));
            }
        }

        let mut result = Value::Object(handled);
        if let Some(root_ref) = self.root_ref.clone()
            && let Value::Object(ref mut map) = result
        {
            map.remove("$ref");
            let definition = if let Some(root_key) = root_ref.strip_prefix("#/$defs/") {
                handled_defs
                    .as_ref()
                    .and_then(|defs| defs.get(root_key))
                    .cloned()
                    .or_else(|| {
                        self.defs
                            .get(root_key)
                            .cloned()
                            .map(|value| self.handle(value))
                    })
            } else if let Some(root_key) = root_ref.strip_prefix("#/definitions/") {
                handled_legacy_defs
                    .as_ref()
                    .and_then(|defs| defs.get(root_key))
                    .cloned()
                    .or_else(|| {
                        self.legacy_defs
                            .get(root_key)
                            .cloned()
                            .map(|value| self.handle(value))
                    })
            } else {
                handled_defs
                    .as_ref()
                    .and_then(|defs| defs.get(root_ref.as_str()))
                    .cloned()
                    .or_else(|| {
                        self.defs
                            .get(root_ref.as_str())
                            .cloned()
                            .map(|value| self.handle(value))
                    })
            };
            if let Some(definition) = definition {
                match definition {
                    Value::Object(def_map) => {
                        for (key, value) in def_map {
                            map.insert(key, value);
                        }
                    }
                    other => {
                        map.insert("$ref".to_string(), other);
                    }
                }
            }
        }

        result
    }

    fn handle(&mut self, schema: Value) -> Value {
        match schema {
            Value::Object(mut map) => {
                let schema_type = map
                    .get("type")
                    .and_then(|value| value.as_str())
                    .map(str::to_string);

                if schema_type.as_deref() == Some("object") {
                    map = self.handle_object(map);
                } else if schema_type.as_deref() == Some("array") {
                    map = self.handle_array(map);
                } else if schema_type.is_none() {
                    map = self.handle_union(map, "anyOf");
                    map = self.handle_union(map, "oneOf");
                }

                map = self.transform(map);
                Value::Object(map)
            }
            Value::Array(items) => {
                Value::Array(items.into_iter().map(|item| self.handle(item)).collect())
            }
            other => other,
        }
    }

    fn handle_object(&mut self, mut schema: Map<String, Value>) -> Map<String, Value> {
        if let Some(Value::Object(properties)) = schema.remove("properties") {
            let mut handled_properties = Map::new();
            for (key, value) in properties {
                handled_properties.insert(key, self.handle(value));
            }
            schema.insert("properties".to_string(), Value::Object(handled_properties));
        }

        if let Some(additional_properties) = schema.remove("additionalProperties") {
            let handled = match additional_properties {
                Value::Bool(value) => Value::Bool(value),
                other => self.handle(other),
            };
            schema.insert("additionalProperties".to_string(), handled);
        }

        if let Some(pattern_properties) = schema.remove("patternProperties") {
            match pattern_properties {
                Value::Object(patterns) => {
                    let mut handled_patterns = Map::new();
                    for (key, value) in patterns {
                        handled_patterns.insert(key, self.handle(value));
                    }
                    schema.insert(
                        "patternProperties".to_string(),
                        Value::Object(handled_patterns),
                    );
                }
                other => {
                    schema.insert("patternProperties".to_string(), other);
                }
            }
        }

        schema
    }

    fn handle_array(&mut self, mut schema: Map<String, Value>) -> Map<String, Value> {
        if let Some(Value::Array(prefix_items)) = schema.remove("prefixItems") {
            let handled = prefix_items
                .into_iter()
                .map(|item| self.handle(item))
                .collect();
            schema.insert("prefixItems".to_string(), Value::Array(handled));
        }

        if let Some(items) = schema.remove("items") {
            schema.insert("items".to_string(), self.handle(items));
        }

        schema
    }

    fn handle_union(
        &mut self,
        mut schema: Map<String, Value>,
        union_key: &str,
    ) -> Map<String, Value> {
        let members = match schema.remove(union_key) {
            Some(Value::Array(members)) => members,
            Some(other) => {
                schema.insert(union_key.to_string(), other);
                return schema;
            }
            None => return schema,
        };

        let handled_members: Vec<Value> =
            members.into_iter().map(|item| self.handle(item)).collect();

        if handled_members.len() == 1
            && let Some(Value::Object(mut only)) = handled_members.first().cloned()
        {
            for (key, value) in schema {
                only.insert(key, value);
            }
            return only;
        }

        schema.insert(union_key.to_string(), Value::Array(handled_members));
        schema
    }

    fn transform(&mut self, mut schema: Map<String, Value>) -> Map<String, Value> {
        schema.remove("title");
        schema.remove("$schema");
        schema.remove("discriminator");

        if schema.contains_key("default") {
            match self.strict {
                Some(true) => {
                    schema.remove("default");
                }
                None => {
                    self.is_strict_compatible = false;
                }
                Some(false) => {}
            }
        }

        if let Some(Value::String(mut schema_ref)) = schema.get("$ref").cloned() {
            if let Some(root_ref) = self.root_ref.as_deref()
                && schema_ref == root_ref
            {
                schema_ref = "#".to_string();
                schema.insert("$ref".to_string(), Value::String(schema_ref.clone()));
            }
            if schema.len() > 1 {
                schema.remove("$ref");
                let mut ref_obj = Map::new();
                ref_obj.insert("$ref".to_string(), Value::String(schema_ref));
                schema.insert(
                    "anyOf".to_string(),
                    Value::Array(vec![Value::Object(ref_obj)]),
                );
            }
        }

        let mut notes = Vec::new();
        for key in STRICT_INCOMPATIBLE_KEYS {
            if let Some(value) = schema.get(key).cloned() {
                match self.strict {
                    Some(true) => {
                        schema.remove(key);
                        notes.push(format!("{key}={}", value_to_string(&value)));
                    }
                    None => {
                        self.is_strict_compatible = false;
                    }
                    Some(false) => {}
                }
            }
        }

        if let Some(Value::String(format)) = schema.get("format").cloned()
            && !STRICT_COMPATIBLE_STRING_FORMATS.contains(&format.as_str())
        {
            match self.strict {
                Some(true) => {
                    schema.remove("format");
                    notes.push(format!("format={format}"));
                }
                None => {
                    self.is_strict_compatible = false;
                }
                Some(false) => {}
            }
        }

        if !notes.is_empty() && self.strict == Some(true) {
            let notes_string = notes.join(", ");
            let description = schema
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            let combined = match description {
                Some(desc) => format!("{desc} ({notes_string})"),
                None => notes_string,
            };
            schema.insert("description".to_string(), Value::String(combined));
        }

        if schema.contains_key("oneOf") {
            if self.strict == Some(true) {
                if let Some(one_of) = schema.remove("oneOf") {
                    schema.insert("anyOf".to_string(), one_of);
                }
            } else if self.strict.is_none() {
                self.is_strict_compatible = false;
            }
        }

        if schema
            .get("type")
            .and_then(|value| value.as_str())
            .is_some_and(|value| value == "object")
        {
            if !schema.contains_key("properties") {
                schema.insert("properties".to_string(), Value::Object(Map::new()));
            }

            match self.strict {
                Some(true) => {
                    schema.insert("additionalProperties".to_string(), Value::Bool(false));
                    let required = schema
                        .get("properties")
                        .and_then(|value| value.as_object())
                        .map(|props| props.keys().cloned().map(Value::String).collect::<Vec<_>>())
                        .unwrap_or_default();
                    schema.insert("required".to_string(), Value::Array(required));
                }
                None => {
                    match schema.get("additionalProperties") {
                        None => {
                            schema.insert("additionalProperties".to_string(), Value::Bool(false));
                        }
                        Some(Value::Bool(false)) => {}
                        Some(_) => {
                            self.is_strict_compatible = false;
                        }
                    }

                    if let (Some(Value::Object(properties)), Some(Value::Array(required))) =
                        (schema.get("properties"), schema.get("required"))
                    {
                        for key in properties.keys() {
                            if !required.iter().any(|value| value.as_str() == Some(key)) {
                                self.is_strict_compatible = false;
                                break;
                            }
                        }
                    } else {
                        self.is_strict_compatible = false;
                    }
                }
                Some(false) => {}
            }
        }

        schema
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_object_sets_required_and_disallows_additional_properties() {
        let schema = json!({
            "title": "Widget",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"}
            }
        });

        let (output, _strict_ok) = transform_openai_schema(&schema, Some(true));
        let obj = output.as_object().expect("schema object");
        assert!(!obj.contains_key("title"));
        assert_eq!(obj.get("additionalProperties"), Some(&Value::Bool(false)));
        let required = obj
            .get("required")
            .and_then(|value| value.as_array())
            .expect("required array");
        assert!(required.contains(&Value::String("name".to_string())));
        assert!(required.contains(&Value::String("count".to_string())));
    }

    #[test]
    fn strict_oneof_is_converted_to_anyof() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            }
        });

        let (output, _strict_ok) = transform_openai_schema(&schema, Some(true));
        let props = output
            .get("properties")
            .and_then(|value| value.as_object())
            .expect("properties");
        let value_schema = props
            .get("value")
            .and_then(|value| value.as_object())
            .expect("value schema");
        assert!(value_schema.contains_key("anyOf"));
        assert!(!value_schema.contains_key("oneOf"));
    }

    #[test]
    fn strict_definitions_are_transformed() {
        let schema = json!({
            "$ref": "#/definitions/Widget",
            "definitions": {
                "Widget": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        });

        let (output, _strict_ok) = transform_openai_schema(&schema, Some(true));
        let obj = output.as_object().expect("schema object");
        assert_eq!(obj.get("type"), Some(&Value::String("object".to_string())));
        assert_eq!(obj.get("additionalProperties"), Some(&Value::Bool(false)));
        let props = obj
            .get("properties")
            .and_then(|value| value.as_object())
            .expect("properties");
        assert!(props.contains_key("name"));
    }

    #[test]
    fn strict_root_enforces_additional_properties() {
        let schema = json!({
            "properties": {
                "name": {"type": "string"}
            }
        });

        let (output, _strict_ok) = transform_openai_schema(&schema, Some(true));
        let obj = output.as_object().expect("schema object");
        assert_eq!(obj.get("type"), Some(&Value::String("object".to_string())));
        assert_eq!(obj.get("additionalProperties"), Some(&Value::Bool(false)));
        let required = obj
            .get("required")
            .and_then(|value| value.as_array())
            .expect("required array");
        assert!(required.contains(&Value::String("name".to_string())));
    }
}
