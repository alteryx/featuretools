"""
Example: Representing a many-to-many relationship in Featuretools
-----------------------------------------------------------------

In this example, we model a university scenario where students can enroll
in multiple courses, and each course can have multiple students.
We handle this many-to-many relationship using an association (bridge) table
called `enrollments` and show how to represent it in a Featuretools
EntitySet by creating two one-to-many relationships that together model
the many-to-many relationship.

This example is intended to be added to the Featuretools docs/examples
directory and is ready to run as-is.
"""

import pandas as pd
import featuretools as ft


# -----------------------------
# 1) Create example dataframes
# -----------------------------
# Students dataframe: one row per student
students_df = pd.DataFrame(
    {
        "student_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

# Courses dataframe: one row per course
courses_df = pd.DataFrame(
    {
        "course_id": [101, 102],
        "title": ["Math", "Science"],
    }
)

# Enrollments association (bridge) table: one row per enrollment
# This table contains foreign keys to both students and courses, and it is
# what allows us to model a many-to-many relationship between students
# and courses.
enrollments_df = pd.DataFrame(
    {
        "enrollment_id": [1, 2, 3, 4],
        "student_id": [1, 1, 2, 3],  # student 1 in two courses
        "course_id": [101, 102, 101, 102],
    }
)


# ---------------------------------
# 2) Build the EntitySet and add data
# ---------------------------------
# Create an empty EntitySet for the university domain
es = ft.EntitySet(id="university")

# Add the students dataframe. Use `student_id` as the index for the
# `students` dataframe.
es = es.add_dataframe(
    dataframe_name="students",
    dataframe=students_df,
    index="student_id",
)

# Add the courses dataframe. Use `course_id` as the index for the
# `courses` dataframe.
es = es.add_dataframe(
    dataframe_name="courses",
    dataframe=courses_df,
    index="course_id",
)

# Add the enrollments dataframe (association/bridge table). Use
# `enrollment_id` as the unique index for enrollments.
es = es.add_dataframe(
    dataframe_name="enrollments",
    dataframe=enrollments_df,
    index="enrollment_id",
)


# ----------------------------------------
# 3) Define relationships to model many-to-many
# ----------------------------------------
# We create two one-to-many relationships:
#  - students -> enrollments (a student can have many enrollments)
#  - courses  -> enrollments (a course can have many enrollments)
# Together, these relationships represent a many-to-many relationship
# between students and courses via the enrollments association table.

relationships = [
    # (parent_df, parent_index, child_df, child_foreign_key)
    ("students", "student_id", "enrollments", "student_id"),
    ("courses", "course_id", "enrollments", "course_id"),
]

# Add relationships to the EntitySet
es = es.add_relationships(relationships)


# ---------------------------------
# 4) Inspect and print the EntitySet
# ---------------------------------
# Print the EntitySet structure so readers can see the entities and
# relationships that model the many-to-many connection.
print(es)

# For extra clarity, show the individual dataframes (optional)
print("\nStudents:\n", es["students"].head())
print("\nCourses:\n", es["courses"].head())
print("\nEnrollments:\n", es["enrollments"].head())
