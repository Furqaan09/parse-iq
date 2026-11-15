"""rename storage_path to storage_key

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12 10:44:59.213236

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename the column and turn old paths into keys."""
    op.alter_column("documents", "storage_path", new_column_name="storage_key")
    # 'storage/uploads/2025/09/x.pdf' -> '2025/09/x.pdf'
    op.execute(
        "UPDATE documents SET storage_key = regexp_replace(storage_key, '^storage/uploads/', '')"
    )


def downgrade() -> None:
    """Turn keys back into paths relative to api/, then rename back."""
    op.execute("UPDATE documents SET storage_key = 'storage/uploads/' || storage_key")
    op.alter_column("documents", "storage_key", new_column_name="storage_path")
